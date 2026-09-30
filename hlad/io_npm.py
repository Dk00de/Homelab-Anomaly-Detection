"""Parser de los access logs de Nginx Proxy Manager (formato por defecto de NPM).

Ejemplo de línea:
[29/Sep/2026:22:58:49 +0000] - 200 200 - GET https tasks.dkoodelab.com "/api/v1/user"
[Client TS_1] [Length 529] [Gzip -] [Sent-to vikunja] "Mozilla/5.0 ..." "https://..."
"""
import re
import warnings
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

# upstream_status puede ser "-", "200" o una lista ("502, 502") si NPM reintentó.
LINE_RE = re.compile(
    r'^\[(?P<ts>[^\]]+)\] - (?P<upstream_status>.+?) (?P<status>\d{3}) - '
    r'(?P<method>\S+) (?P<scheme>\S+) (?P<host>\S+) "(?P<url>[^"]*)" '
    r'\[Client (?P<client>[^\]]+)\] \[Length (?P<length>\d+)\] \[Gzip [^\]]*\] '
    r'\[Sent-to (?P<upstream>[^\]]+)\] "(?P<user_agent>[^"]*)" "(?P<referer>[^"]*)"'
)


def parse_npm_log(path: Path) -> pd.DataFrame:
    """Devuelve un DataFrame con una fila por petición.

    Columnas: ts, method, scheme, host, path, query, body, status, length,
    client, upstream, user_agent, referer. `body` va vacío: NPM no lo registra.
    """
    rows, unparsed = [], 0
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = LINE_RE.match(line)
            if m:
                rows.append(m.groupdict())
            elif line.strip():
                unparsed += 1
    if unparsed:
        warnings.warn(f"{path.name}: {unparsed} líneas no reconocidas")

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], format="%d/%b/%Y:%H:%M:%S %z", utc=True)
    parts = df["url"].map(urlsplit)
    df["path"] = parts.map(lambda p: p.path)
    df["query"] = parts.map(lambda p: p.query)
    df["body"] = ""
    df["status"] = df["status"].astype(int)
    df["length"] = df["length"].astype(int)
    df = df.drop(columns=["url", "upstream_status"])
    return df.sort_values("ts", ignore_index=True)
