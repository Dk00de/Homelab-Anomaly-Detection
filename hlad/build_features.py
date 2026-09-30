"""Genera data/processed/ a partir de data/raw/.

Uso: python -m hlad.build_features
"""
import pandas as pd

from .features import FEATURE_COLUMNS, extract_features
from .io_csic import load_csic
from .io_npm import parse_npm_log
from .paths import CSIC_CSV, DATA_PROCESSED, NPM_ACCESS_LOG


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    csic = load_csic(CSIC_CSV)
    csic_x = extract_features(csic)
    csic_out = pd.concat([csic_x, csic[["label"]]], axis=1)
    csic_out.to_csv(DATA_PROCESSED / "csic_features.csv", index=False)

    npm = parse_npm_log(NPM_ACCESS_LOG)
    npm_x = extract_features(npm)
    # Se conservan ts y host como metadatos para el generador sintético; no son variables del modelo.
    npm_out = pd.concat([npm[["ts", "host", "method", "path", "query"]], npm_x], axis=1)
    npm_out["label"] = 0
    npm_out.to_csv(DATA_PROCESSED / "homelab_normal_features.csv", index=False)

    print(f"CSIC:    {len(csic_out):>6} filas  ({csic_out['label'].sum()} anómalas)")
    print(f"Homelab: {len(npm_out):>6} filas  (todas normales)")

    means = pd.DataFrame({
        "csic_normal": csic_x[csic["label"] == 0].mean(),
        "csic_anomalo": csic_x[csic["label"] == 1].mean(),
        "homelab": npm_x.mean(),
    }).loc[FEATURE_COLUMNS]
    print("\nMedias por variable:")
    print(means.round(3).to_string())

    tok = [c for c in FEATURE_COLUMNS if c.startswith("tok_")]
    print("\n% de filas con algún token de ataque:")
    print(f"  CSIC normal:  {(csic_x.loc[csic['label'] == 0, tok].sum(axis=1) > 0).mean():.2%}")
    print(f"  CSIC anómalo: {(csic_x.loc[csic['label'] == 1, tok].sum(axis=1) > 0).mean():.2%}")
    print(f"  Homelab:      {(npm_x[tok].sum(axis=1) > 0).mean():.2%}")


if __name__ == "__main__":
    main()
