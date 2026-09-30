"""Extracción de variables por petición HTTP.

Dos grupos:
- FEATURE_COLUMNS (contenido): solo campos presentes en CSIC y en el homelab
  (método, ruta, query, body). Es lo único que usa el experimento A.
- CONTEXT_COLUMNS (contexto): código de respuesta, bytes, tipo de User-Agent y
  hora. Solo existen en el homelab → experimento B (contenido + contexto).

Se excluyen a propósito el host/servicio y la IP (esta última es el gateway de
Docker en la mayoría de los logs, ver data/README.md).
"""
import math
import re
from collections import Counter
from urllib.parse import unquote_plus

import numpy as np
import pandas as pd

# Patrones por familia de ataque. Se buscan sobre el payload decodificado y en
# minúsculas, exigiendo contexto (espacios, paréntesis, operadores) para no
# disparar con nombres legítimos como "LanguageSelector.js" o "/api/subscriptions".
TOKEN_PATTERNS = {
    "sqli": [
        r"\bunion\b[\s(]+(all\s+)?select\b", r"\bselect\b.+\bfrom\b", r"\binsert\s+into\b",
        r"\bdrop\s+table\b", r"\bdelete\s+from\b", r"\bupdate\s+\w+\s+set\b",
        r"'\s*(or|and)\s", r"\b(or|and)\s+\d+\s*=\s*\d+", r"--\s*$", r"/\*.*\*/", r";\s*--",
        r"\b(sleep|benchmark|waitfor\s+delay|pg_sleep)\s*\(", r"\bxp_cmdshell\b",
        r"\binformation_schema\b", r"@@version",
    ],
    "xss": [
        r"<\s*script", r"javascript\s*:", r"\bon(error|load|mouseover|click|focus)\s*=",
        r"\balert\s*\(", r"<\s*(img|iframe|svg|body|object|embed)\b", r"document\.(cookie|location)",
        r"\beval\s*\(",
    ],
    "traversal": [
        r"\.\./", r"\.\.\\", r"/etc/(passwd|shadow|hosts)", r"\b(win|boot|system)\.ini\b",
        r"/proc/self/", r"%00",
    ],
    "cmdi": [
        r";\s*(ls|cat|id|whoami|uname|wget|curl|nc|bash|sh)\b", r"\|\s*(ls|cat|id|whoami|nc|bash|sh)\b",
        r"`[^`]+`", r"\$\([^)]+\)", r"&&\s*\w+",
    ],
    "sensitive": [
        r"/\.env\b", r"/\.git\b", r"/\.htaccess", r"\bwp-(login|admin|content|includes)\b",
        r"\bphpmyadmin\b", r"\.(bak|old|orig|swp|sql|inc|config)\b", r"\bcgi-bin\b",
        r"\bshell\.php\b", r"\bxmlrpc\.php\b",
    ],
}
COMPILED = {fam: [re.compile(p) for p in pats] for fam, pats in TOKEN_PATTERNS.items()}

PCT_RE = re.compile(r"%[0-9a-fA-F]{2}")
SPECIAL_CHARS = "'\"<>;()`|\\{}[]$*!@^~"

FEATURE_COLUMNS = [
    "method_get", "method_post", "method_other",
    "path_len", "path_depth", "query_len", "n_params", "body_len", "payload_len",
    "n_pct_encoded", "n_quote", "n_angle", "n_semicolon", "n_paren", "n_special",
    "ratio_special", "ratio_digit", "ratio_upper", "entropy", "max_param_len",
    "tok_sqli", "tok_xss", "tok_traversal", "tok_cmdi", "tok_sensitive",
]


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def _n_params(q: str) -> int:
    return len([p for p in q.split("&") if p]) if q else 0


def _max_param_len(q: str, body: str) -> int:
    vals = [p.split("=", 1)[1] if "=" in p else p
            for part in (q, body) if part for p in part.split("&")]
    return max((len(v) for v in vals), default=0)


def _decode(s: str) -> str:
    # Doble decodificación: los ataques a menudo vienen codificados dos veces (%2527).
    return unquote_plus(unquote_plus(s))


def extract_one(method: str, path: str, query: str, body: str) -> dict:
    raw_payload = f"{path}?{query}" if query else path
    if body:
        raw_payload += f"\n{body}"
    dec = _decode(raw_payload)
    low = dec.lower()
    n = max(len(dec), 1)
    m = method.upper()

    feats = {
        "method_get": int(m == "GET"),
        "method_post": int(m == "POST"),
        "method_other": int(m not in ("GET", "POST")),
        "path_len": len(path),
        "path_depth": path.count("/"),
        "query_len": len(query),
        "n_params": _n_params(query) + _n_params(body),
        "body_len": len(body),
        "payload_len": len(raw_payload),
        "n_pct_encoded": len(PCT_RE.findall(raw_payload)),
        "n_quote": dec.count("'") + dec.count('"'),
        "n_angle": dec.count("<") + dec.count(">"),
        "n_semicolon": dec.count(";"),
        "n_paren": dec.count("(") + dec.count(")"),
        "n_special": sum(dec.count(c) for c in SPECIAL_CHARS),
        "ratio_digit": sum(c.isdigit() for c in dec) / n,
        "ratio_upper": sum(c.isupper() for c in dec) / n,
        "entropy": _entropy(dec),
        "max_param_len": _max_param_len(query, body),
    }
    feats["ratio_special"] = feats["n_special"] / n
    for fam, pats in COMPILED.items():
        feats[f"tok_{fam}"] = sum(bool(p.search(low)) for p in pats)
    return feats


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """df necesita columnas method, path, query, body. Devuelve FEATURE_COLUMNS."""
    rows = [extract_one(m, p, q, b)
            for m, p, q, b in zip(df["method"], df["path"], df["query"], df["body"])]
    return pd.DataFrame(rows, index=df.index)[FEATURE_COLUMNS].astype(np.float64)


# --- Contexto (solo homelab) -------------------------------------------------

# Orden importa: Nikto y zgrab se anuncian como "Mozilla/5.00 (Nikto...)", así que
# se revisan herramientas antes que navegadores. okhttp/Dart no están: son
# librerías de apps legítimas (Vikunja Mobile, CapyReader).
TOOL_UA_RE = re.compile(
    r"curl|wget|python|go-http|sqlmap|nikto|nmap|masscan|zgrab|gobuster|dirbuster|"
    r"ffuf|fuzz|wfuzz|hydra|nuclei|scanner|libwww|httpclient|java/|bot\b|spider|crawl",
    re.I,
)
UA_KINDS = ["browser", "app", "tool", "empty"]

CONTEXT_COLUMNS = [
    "status_2xx", "status_3xx", "status_4xx", "status_5xx",
    "status_401_403", "status_404", "log_length",
    "ua_browser", "ua_app", "ua_tool", "ua_empty",
    "hour_sin", "hour_cos",
]


def ua_kind(ua: str) -> str:
    if not ua or ua == "-":
        return "empty"
    if TOOL_UA_RE.search(ua):
        return "tool"
    if ua.startswith("Mozilla/"):
        return "browser"
    return "app"


def extract_context(df: pd.DataFrame) -> pd.DataFrame:
    """df necesita status, length, user_agent y ts (UTC). Devuelve CONTEXT_COLUMNS."""
    st = df["status"].astype(int)
    kinds = df["user_agent"].map(ua_kind)
    hour = pd.to_datetime(df["ts"], utc=True).dt.hour + pd.to_datetime(df["ts"], utc=True).dt.minute / 60
    out = pd.DataFrame({
        "status_2xx": st.between(200, 299),
        "status_3xx": st.between(300, 399),
        "status_4xx": st.between(400, 499),
        "status_5xx": st.between(500, 599),
        "status_401_403": st.isin([401, 403]),
        "status_404": st == 404,
        "log_length": np.log1p(df["length"].astype(float)),
        **{f"ua_{k}": kinds == k for k in UA_KINDS},
        # Hora como círculo: 23:00 y 00:00 quedan cerca.
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
    }, index=df.index)
    return out[CONTEXT_COLUMNS].astype(np.float64)
