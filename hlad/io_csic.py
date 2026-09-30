"""Carga del CSIC 2010 HTTP Dataset (versión CSV de Kaggle)."""
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd


def load_csic(path: Path) -> pd.DataFrame:
    """Devuelve method, path, query, body y label (0 normal / 1 anómalo).

    La columna URL viene como "http://localhost:8080/tienda1/... HTTP/1.1":
    se quitan esquema, host y versión para quedarse con ruta y query, igual
    que en los logs de NPM.
    """
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    url = raw["URL"].str.replace(r"\s+HTTP/\d\.\d$", "", regex=True)
    parts = url.map(urlsplit)
    return pd.DataFrame({
        "method": raw["Method"],
        "path": parts.map(lambda p: p.path),
        "query": parts.map(lambda p: p.query),
        "body": raw["content"],
        "label": raw["classification"].astype(int),
    })
