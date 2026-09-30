"""Ciclo de experimentos: escenario × modelo × ruido × desbalance × semilla.

Uso: python -m hlad.experiments
Salida: results/runs.csv (una fila por corrida) y resumen en consola.

Escenarios (los tres se evalúan sobre el MISMO conjunto de prueba):
- A            entrenado con CSIC 2010, variables de contenido.
- B_contenido  entrenado con homelab + ataques sintéticos, variables de contenido.
               A vs B_contenido aísla el efecto de los DATOS de entrenamiento.
- B_completo   igual, con contenido + contexto.
               B_contenido vs B_completo aísla el efecto de las VARIABLES.
Además, para cada conjunto de prueba:
- regla/fail2ban          la regla de comportamiento sola (hlad/rules.py).
- <escenario>+fail2ban    modelo OR regla: defensa en capas.

División del tráfico normal: temporal. El 70% más antiguo entrena y el 30% más
reciente prueba, como en un despliegue real (el modelo no ve el futuro).
"""
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .features import CONTEXT_COLUMNS, FEATURE_COLUMNS, extract_context, extract_features
from .io_csic import load_csic
from .io_npm import parse_npm_log
from .models import MODELS
from .paths import CSIC_CSV, NPM_ACCESS_LOG, RESULTS
from .rules import fail2ban_flags
from .synthetic import FAMILY_MIX, AttackGenerator, payload_pool, split_csic_attacks

SEEDS = [0, 1, 2, 3, 4]
NOISES = ["bajo", "medio", "alto"]
RATIOS = [0.5, 0.1, 0.01]
TRAIN_FRACTION = 0.7


def features(df: pd.DataFrame, context: bool) -> pd.DataFrame:
    X = extract_features(df)
    return pd.concat([X, extract_context(df)], axis=1) if context else X


def with_attacks(normal: pd.DataFrame, X_normal: pd.DataFrame, payloads: pd.DataFrame,
                 ratio: float, noise: str, seed: int):
    """Normal + ataques en la proporción pedida. Reutiliza las variables ya calculadas del normal."""
    n_att = round(ratio / (1 - ratio) * len(normal))
    att = AttackGenerator(normal, payloads, noise, seed).generate(n_att)
    df = pd.concat([normal.assign(label=0, attack_type="normal", session_id=-1), att], ignore_index=True)
    X = pd.concat([X_normal, features(att, context=True)], ignore_index=True)
    return df, X


def metrics(y: np.ndarray, pred: np.ndarray, score, attack_type: np.ndarray) -> dict:
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    out = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "pr_auc": average_precision_score(y, score) if score is not None else np.nan,
    }
    for fam in FAMILY_MIX:
        m = attack_type == fam
        out[f"recall_{fam}"] = pred[m].mean() if m.any() else np.nan
    return out


def main() -> None:
    warnings.simplefilter("ignore")
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)

    normal = parse_npm_log(NPM_ACCESS_LOG)
    cut = normal["ts"].quantile(TRAIN_FRACTION)
    train_n = normal[normal["ts"] < cut].reset_index(drop=True)
    test_n = normal[normal["ts"] >= cut].reset_index(drop=True)
    X_train_n = features(train_n, context=True)
    X_test_n = features(test_n, context=True)
    print(f"Normal: {len(train_n)} entrenamiento (hasta {cut:%Y-%m-%d %H:%M}) / {len(test_n)} prueba")

    csic = load_csic(CSIC_CSV)
    csic_normal = csic[csic["label"] == 0]

    rows = []
    for seed in SEEDS:
        pools = split_csic_attacks(csic, seed)
        pay_train = payload_pool(pools["gen_train"], csic_normal)
        pay_test = payload_pool(pools["gen_test"], csic_normal)

        # Experimento A: se entrena una vez por semilla (no depende de ruido ni desbalance).
        a_train = pd.concat([csic_normal, pools["A_train"]], ignore_index=True)
        XA, yA = extract_features(a_train), a_train["label"].to_numpy()
        a_models = [M(seed).fit(XA, yA) for M in MODELS]

        for ni, noise in enumerate(NOISES):
            for ri, ratio in enumerate(RATIOS):
                gs = seed * 100 + ni * 10 + ri
                test, X_test = with_attacks(test_n, X_test_n, pay_test, ratio, noise, gs)
                train, X_train = with_attacks(train_n, X_train_n, pay_train, ratio, noise, gs + 50_000)
                y_te, y_tr = test["label"].to_numpy(), train["label"].to_numpy()
                fam = test["attack_type"].to_numpy()
                f2b = fail2ban_flags(test)
                key = {"seed": seed, "ruido": noise, "ataques": ratio}

                rows.append({**key, "escenario": "regla", "modelo": "fail2ban",
                             **metrics(y_te, f2b, None, fam)})

                scenarios = {
                    "A": (a_models, FEATURE_COLUMNS),
                    "B_contenido": ([M(seed).fit(X_train[FEATURE_COLUMNS], y_tr) for M in MODELS], FEATURE_COLUMNS),
                    "B_completo": ([M(seed).fit(X_train, y_tr) for M in MODELS], FEATURE_COLUMNS + CONTEXT_COLUMNS),
                }
                for scen, (models, cols) in scenarios.items():
                    for m in models:
                        score = m.score(X_test[cols])
                        pred = score >= m.threshold
                        rows.append({**key, "escenario": scen, "modelo": m.name,
                                     **metrics(y_te, pred, score, fam)})
                        rows.append({**key, "escenario": f"{scen}+fail2ban", "modelo": m.name,
                                     **metrics(y_te, pred | f2b, None, fam)})
        print(f"  semilla {seed} lista ({time.time() - t0:.0f}s)")

    runs = pd.DataFrame(rows)
    runs.to_csv(RESULTS / "runs.csv", index=False)
    print(f"\n{len(runs)} corridas → {RESULTS / 'runs.csv'} ({time.time() - t0:.0f}s)")

    pd.set_option("display.width", 220)
    base = runs[~runs["escenario"].str.endswith("+fail2ban")]
    print("\n== F1 medio por escenario y modelo (filas) × ruido (columnas), 10% de ataques")
    print(base[base["ataques"] == 0.1].pivot_table(index=["escenario", "modelo"], columns="ruido",
                                                   values="f1").reindex(columns=NOISES).round(3).to_string())
    print("\n== F1 medio × desbalance, ruido medio")
    print(base[base["ruido"] == "medio"].pivot_table(index=["escenario", "modelo"], columns="ataques",
                                                     values="f1").round(3).to_string())
    print("\n== Recall por familia, ruido medio, 10% ataques")
    cols = [f"recall_{f}" for f in FAMILY_MIX] + ["recall", "precision", "fpr"]
    sel = runs[(runs["ruido"] == "medio") & (runs["ataques"] == 0.1)]
    print(sel.groupby(["escenario", "modelo"])[cols].mean().round(3).to_string())


if __name__ == "__main__":
    main()
