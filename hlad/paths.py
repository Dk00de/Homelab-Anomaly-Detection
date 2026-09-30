"""Rutas del proyecto."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

CSIC_CSV = DATA_RAW / "csic_database.csv"
NPM_ACCESS_LOG = DATA_RAW / "npm_access.log"
