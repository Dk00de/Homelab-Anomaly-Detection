"""Cuantifica qué tan distintos son CSIC 2010 y el tráfico del homelab.

Uso: python -m hlad.compare_datasets
Salida: results/dataset_comparison.csv y resumen en consola.

Tres preguntas:
1. Por variable: ¿cambia la distribución entre CSIC normal y homelab? (KS, diferencia
   de medias estandarizada, % de ceros).
2. En conjunto: ¿un clasificador distingue de qué dataset viene una petición normal?
   (AUC del "clasificador de dominio"; 0.5 = indistinguibles, 1.0 = totalmente distintos).
3. Concentración: ¿cuántas variables usa realmente un árbol entrenado con CSIC?
"""
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier

from .features import FEATURE_COLUMNS
from .paths import DATA_PROCESSED, RESULTS

SEED = 42


def per_feature(csic_n: pd.DataFrame, csic_a: pd.DataFrame, home: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in FEATURE_COLUMNS:
        pooled_sd = np.sqrt((csic_n[c].var() + home[c].var()) / 2) or np.nan
        rows.append({
            "variable": c,
            "ks_csicN_vs_home": ks_2samp(csic_n[c], home[c]).statistic,
            "smd_csicN_vs_home": (home[c].mean() - csic_n[c].mean()) / pooled_sd,
            "ks_csicN_vs_csicA": ks_2samp(csic_n[c], csic_a[c]).statistic,
            "ceros_csicN": (csic_n[c] == 0).mean(),
            "ceros_csicA": (csic_a[c] == 0).mean(),
            "ceros_home": (home[c] == 0).mean(),
            "valores_distintos_home": home[c].nunique(),
        })
    return pd.DataFrame(rows).set_index("variable")


def domain_classifier(csic_n: pd.DataFrame, home: pd.DataFrame) -> tuple[float, pd.Series]:
    n = min(len(csic_n), len(home))
    X = pd.concat([csic_n.sample(n, random_state=SEED), home.sample(n, random_state=SEED)])
    y = np.r_[np.zeros(n), np.ones(n)]
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=SEED, n_jobs=-1)
    auc = cross_val_score(rf, X, y, cv=5, scoring="roc_auc").mean()
    rf.fit(X, y)
    return auc, pd.Series(rf.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)


def tree_concentration(csic: pd.DataFrame) -> pd.Series:
    X, y = csic[FEATURE_COLUMNS], csic["label"]
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEED)
    tree = DecisionTreeClassifier(max_depth=10, random_state=SEED).fit(Xtr, ytr)
    return pd.Series(tree.feature_importances_, index=FEATURE_COLUMNS).sort_values(ascending=False)


def n_for_share(imp: pd.Series, share: float) -> int:
    return int((imp.cumsum() < share).sum() + 1)


def main() -> None:
    csic = pd.read_csv(DATA_PROCESSED / "csic_features.csv")
    home = pd.read_csv(DATA_PROCESSED / "homelab_normal_features.csv")[FEATURE_COLUMNS]
    csic_n = csic.loc[csic["label"] == 0, FEATURE_COLUMNS]
    csic_a = csic.loc[csic["label"] == 1, FEATURE_COLUMNS]

    pf = per_feature(csic_n, csic_a, home)
    RESULTS.mkdir(exist_ok=True)
    pf.round(4).to_csv(RESULTS / "dataset_comparison.csv")

    pd.set_option("display.width", 200)
    print("== Por variable (KS: 0 = misma distribución, 1 = disjuntas)")
    print(pf.sort_values("ks_csicN_vs_home", ascending=False).round(3).to_string())

    muertas = pf.index[(pf["ceros_home"] > 0.99) | (pf["valores_distintos_home"] <= 1)].tolist()
    print(f"\nVariables casi siempre en cero en el homelab (>99%): {len(muertas)} → {muertas}")
    informativas = pf.index[pf["ks_csicN_vs_csicA"] > 0.1].tolist()
    print(f"Variables que separan normal/anómalo en CSIC (KS > 0.1): {len(informativas)} → {informativas}")

    auc, dom_imp = domain_classifier(csic_n, home)
    print(f"\n== Clasificador de dominio (CSIC normal vs homelab): AUC = {auc:.3f}")
    print("Variables que más delatan el dominio:")
    print(dom_imp.head(8).round(3).to_string())

    imp = tree_concentration(csic)
    print("\n== Árbol entrenado en CSIC: importancia de variables")
    print(imp[imp > 0].round(3).to_string())
    print(f"Variables con importancia > 0: {(imp > 0).sum()} de {len(imp)}")
    print(f"Variables para el 80% / 90% de la importancia: {n_for_share(imp, .8)} / {n_for_share(imp, .9)}")


if __name__ == "__main__":
    main()
