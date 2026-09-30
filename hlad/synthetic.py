"""Generador de ataques sintéticos sobre el tráfico real del homelab.

El homelab no registró ningún ataque real (solo es accesible por Tailscale), así
que las peticiones maliciosas se generan con el mismo esquema que parse_npm_log:
ts, method, host, path, query, body, status, length, user_agent...

Tres familias:
- inyeccion:     un payload anómalo de CSIC 2010 insertado en una ruta REAL del
                 homelab, reemplazando el valor de uno de sus parámetros.
- escaneo:       rutas típicas de reconocimiento automatizado (/.env, /wp-login.php…).
- fuerza_bruta:  ráfagas contra endpoints de login que existen en el homelab.

Dos factores del experimento:
- ruido (bajo / medio / alto): qué tanto se camufla el ataque como tráfico normal
  (UA de navegador real, código de respuesta y horario como los normales, payloads
  sin caracteres ni palabras sospechosas). Más ruido = más solapamiento entre clases.
- desbalance: proporción de ataques en el dataset (ver build_dataset).

Fuga de información: los ataques anómalos de CSIC se dividen en tres grupos
disjuntos (split_csic_attacks) para que el experimento A no entrene con los mismos
payloads con los que después se prueba.
"""
import re
from dataclasses import dataclass
from urllib.parse import unquote_plus

import numpy as np
import pandas as pd

from .features import extract_one, ua_kind

FAMILY_MIX = {"inyeccion": 0.60, "escaneo": 0.25, "fuerza_bruta": 0.15}


@dataclass(frozen=True)
class Noise:
    p_browser_ua: float      # el ataque usa un UA de navegador tomado del tráfico real
    p_status_normal: float   # el código de respuesta se toma de la distribución normal
    p_hour_normal: float     # la hora se toma del horario real del homelab (si no: uniforme)
    p_quiet_payload: float   # el payload viene del subconjunto "silencioso" de CSIC


NOISE_LEVELS = {
    "bajo": Noise(0.10, 0.10, 0.10, 0.10),
    "medio": Noise(0.50, 0.50, 0.50, 0.50),
    "alto": Noise(0.90, 0.90, 0.90, 0.90),
}

# Rutas de reconocimiento conocidas públicamente (listas de escáneres comunes).
# Incluye algunas inocuas (robots.txt, favicon.ico) porque los escáneres también
# las piden: solapamiento natural con tráfico normal.
SCAN_PATHS = [
    "/.env", "/.env.bak", "/.git/config", "/.git/HEAD", "/wp-login.php", "/wp-admin/",
    "/xmlrpc.php", "/phpmyadmin/", "/pma/", "/admin", "/admin/login", "/administrator/",
    "/config.php", "/config.json", "/server-status", "/.DS_Store", "/backup.zip",
    "/backup.sql", "/db.sql", "/.aws/credentials", "/actuator/env", "/actuator/health",
    "/cgi-bin/luci", "/HNAP1/", "/owa/auth/logon.aspx", "/solr/admin/info/system",
    "/console", "/manager/html", "/_ignition/health-check", "/telescope/requests",
    "/index.php", "/login.php", "/shell.php", "/.well-known/security.txt",
    "/robots.txt", "/sitemap.xml", "/favicon.ico",
]

# Endpoints de login observados en los logs reales (+ Vikunja y RomM, cuya API es pública).
LOGIN_ENDPOINTS = [
    ("music.dkoodelab.com", "POST", "/auth/login"),
    ("books.dkoodelab.com", "POST", "/api/v1/auth/login"),
    ("npm.dkoodelab.com", "POST", "/api/tokens"),
    ("wall.dkoodelab.com", "POST", "/oauth/v2/token"),
    ("tasks.dkoodelab.com", "POST", "/api/v1/login"),
    ("rom.dkoodelab.com", "POST", "/api/login"),
]

TOOL_UAS = [
    "sqlmap/1.8.2#stable (https://sqlmap.org)",
    "Mozilla/5.00 (Nikto/2.5.0) (Evasions:None) (Test:000001)",
    "python-requests/2.32.3",
    "curl/8.5.0",
    "Go-http-client/1.1",
    "Mozilla/5.0 zgrab/0.x",
    "gobuster/3.6",
    "Fuzz Faster U Fool v2.1.0-dev",
    "Mozilla/4.0 (Hydra)",
    "",
]

# Nombres de parámetro reales del homelab, para inyectar donde la ruta no tenía query.
PARAM_NAMES = ["id", "s", "page", "size", "filter", "q", "search", "f", "c", "v"]

ATTACK_STATUS = {
    "inyeccion": {200: .35, 400: .25, 500: .20, 404: .15, 403: .05},
    "escaneo": {404: .80, 403: .10, 301: .05, 200: .05},
    "fuerza_bruta": {401: .85, 400: .05, 403: .05, 429: .05},
}
SESSION_SIZE = {"inyeccion": (5, 50), "escaneo": (20, 150), "fuerza_bruta": (20, 200)}
SESSION_GAP_S = {"inyeccion": 2.0, "escaneo": 0.5, "fuerza_bruta": 1.0}


# --- Payloads de CSIC --------------------------------------------------------

def _params(s: str) -> list[tuple[str, str]]:
    return [tuple(p.split("=", 1)) if "=" in p else (p, "") for p in s.split("&") if p]


PLAIN_RE = re.compile(r"[\w\s.,/@:+\-]*")
QUIET_MAX_LOUDNESS = 5


def _loudness(value: str) -> int:
    """Qué tan evidente es un payload: tokens de ataque pesan más que caracteres sueltos."""
    f = extract_one("GET", "/", f"x={value}", "")
    tokens = sum(v for k, v in f.items() if k.startswith("tok_"))
    return int(5 * tokens + f["n_special"] + f["n_pct_encoded"])


def payload_pool(csic_attacks: pd.DataFrame, csic_normal: pd.DataFrame) -> pd.DataFrame:
    """Extrae de cada petición anómala de CSIC el valor de parámetro anómalo.

    Una petición anómala de CSIC suele traer varios parámetros normales y uno
    alterado. Solo cuenta como candidato un par (parámetro, valor) que nunca aparece
    en el tráfico normal de CSIC; entre los candidatos se toma el más sospechoso.
    Si no hay candidatos (la anomalía está en la ruta), la fila se descarta: la
    familia 'escaneo' cubre ese caso.

    Además se descartan las anomalías *semánticas*: valores de texto o números
    planos ("Vaciar carrito", "365", un correo) que solo son anómalos para la
    tienda de CSIC (cantidad fuera de rango, parámetro renombrado…) y que en otra
    aplicación no significan nada. Son ~55% de los candidatos.

    Devuelve columnas value, loudness, quiet:
    - quiet (sutil):   loudness 1–5. Una comilla, un byte nulo, CRLF, caracteres raros.
    - no quiet (claro): loudness >= 6. <script>, waitfor delay, SSI, etc.
    """
    seen = {kv for q, b in zip(csic_normal["query"], csic_normal["body"])
            for kv in _params(q) + _params(b)}
    rows = []
    for q, b in zip(csic_attacks["query"], csic_attacks["body"]):
        vals = [v for k, v in _params(q) + _params(b) if v and (k, v) not in seen]
        # CSIC codifica acentos en latin-1 (%E9 = é): decodificar así para que
        # "Ingl%E9s" cuente como texto plano.
        vals = [v for v in vals
                if not PLAIN_RE.fullmatch(unquote_plus(unquote_plus(v, encoding="latin-1"), encoding="latin-1"))]
        if not vals:
            continue
        score, value = max((_loudness(v), v) for v in vals)
        rows.append({"value": value, "loudness": score, "quiet": score <= QUIET_MAX_LOUDNESS})
    return pd.DataFrame(rows)


def split_csic_attacks(csic: pd.DataFrame, seed: int,
                       shares=(0.4, 0.3, 0.3)) -> dict[str, pd.DataFrame]:
    """Divide las anómalas de CSIC en tres grupos disjuntos.

    A_train:   entrena el experimento A (junto con todas las normales de CSIC).
    gen_train: payloads para los ataques sintéticos del conjunto de entrenamiento de B.
    gen_test:  payloads para los ataques del conjunto de prueba (común a A y B).
    """
    att = csic[csic["label"] == 1].sample(frac=1, random_state=seed)
    a = int(shares[0] * len(att))
    b = a + int(shares[1] * len(att))
    return {"A_train": att.iloc[:a], "gen_train": att.iloc[a:b], "gen_test": att.iloc[b:]}


# --- Generador ---------------------------------------------------------------

class AttackGenerator:
    def __init__(self, normal: pd.DataFrame, payloads: pd.DataFrame, noise: str, seed: int):
        self.rng = np.random.default_rng(seed)
        self.noise = NOISE_LEVELS[noise]
        self.normal = normal.reset_index(drop=True)
        self.loud = payloads.loc[~payloads["quiet"], "value"].to_numpy()
        self.quiet = payloads.loc[payloads["quiet"], "value"].to_numpy()

        self.browser_uas = normal.loc[normal["user_agent"].map(ua_kind) == "browser", "user_agent"].to_numpy()
        self.status_by_method = {m: g.to_numpy() for m, g in normal.groupby("method")["status"]}
        self.len_by_status = {s: g.to_numpy() for s, g in normal.groupby("status")["length"]}
        self.hosts = normal["host"].unique()
        ts = pd.to_datetime(normal["ts"], utc=True)
        self.t0, self.t1 = ts.min(), ts.max()
        self.normal_tod = ((ts - ts.dt.normalize()).dt.total_seconds()).to_numpy()

    # utilidades de muestreo
    def _choice(self, arr):
        return arr[self.rng.integers(len(arr))]

    def _weighted(self, dist: dict):
        keys = list(dist)
        return keys[self.rng.choice(len(keys), p=np.array(list(dist.values())))]

    def _start(self) -> pd.Timestamp:
        day = self.t0.normalize() + pd.Timedelta(days=int(self.rng.integers((self.t1 - self.t0).days + 1)))
        if self.rng.random() < self.noise.p_hour_normal:
            tod = self._choice(self.normal_tod)
        else:
            tod = self.rng.uniform(0, 86400)
        return day + pd.Timedelta(seconds=float(tod))

    def _status(self, family: str, method: str) -> int:
        # Camuflado: el código sale de peticiones normales con el mismo método
        # (un POST de login no devuelve 304).
        normal = self.status_by_method.get(method)
        if normal is not None and self.rng.random() < self.noise.p_status_normal:
            return int(self._choice(normal))
        return int(self._weighted(ATTACK_STATUS[family]))

    def _length(self, status: int) -> int:
        pool = self.len_by_status.get(status)
        return int(self._choice(pool)) if pool is not None else int(self.rng.integers(30, 200))

    def _ua(self) -> str:
        if self.rng.random() < self.noise.p_browser_ua:
            return str(self._choice(self.browser_uas))
        return self._choice(TOOL_UAS)

    def _payload(self) -> str:
        use_quiet = self.rng.random() < self.noise.p_quiet_payload and len(self.quiet)
        return str(self._choice(self.quiet if use_quiet else self.loud))

    # familias
    def _inyeccion(self, _session) -> dict:
        base = self.normal.iloc[self.rng.integers(len(self.normal))]
        value = self._payload()
        params = _params(base["query"])
        if params:
            i = self.rng.integers(len(params))
            params[i] = (params[i][0], value)
            query = "&".join(f"{k}={v}" for k, v in params)
        else:
            query = f"{self._choice(PARAM_NAMES)}={value}"
        return {"host": base["host"], "method": "GET", "path": base["path"], "query": query}

    def _escaneo(self, session) -> dict:
        method = self._weighted({"GET": .85, "HEAD": .10, "POST": .05})
        return {"host": session["host"], "method": method, "path": self._choice(SCAN_PATHS), "query": ""}

    def _fuerza_bruta(self, session) -> dict:
        host, method, path = session["endpoint"]
        return {"host": host, "method": method, "path": path, "query": ""}

    def generate(self, n: int) -> pd.DataFrame:
        rows, sid = [], 0
        families = list(FAMILY_MIX)
        probs = np.array(list(FAMILY_MIX.values()))
        while len(rows) < n:
            fam = families[self.rng.choice(len(families), p=probs)]
            session = {
                "ua": self._ua(),
                "host": self._choice(self.hosts),
                "endpoint": LOGIN_ENDPOINTS[self.rng.integers(len(LOGIN_ENDPOINTS))],
            }
            lo, hi = SESSION_SIZE[fam]
            k = int(self.rng.integers(lo, hi + 1))
            gaps = self.rng.exponential(SESSION_GAP_S[fam], size=k).cumsum()
            start = self._start()
            build = getattr(self, f"_{fam}")
            for g in gaps:
                req = build(session)
                status = self._status(fam, req["method"])
                rows.append({
                    "ts": start + pd.Timedelta(seconds=float(g)),
                    "scheme": "https", "body": "", "client": "LAN_2", "upstream": "-", "referer": "-",
                    "status": status, "length": self._length(status), "user_agent": session["ua"],
                    "label": 1, "attack_type": fam, "session_id": sid,
                    **req,
                })
            sid += 1
        return pd.DataFrame(rows[:n])


def build_dataset(normal: pd.DataFrame, payloads: pd.DataFrame, attack_ratio: float,
                  noise: str, seed: int) -> pd.DataFrame:
    """Tráfico normal real + ataques sintéticos en la proporción pedida.

    attack_ratio es la fracción de ataques en el resultado (p. ej. 0.5, 0.1, 0.01).
    """
    n_att = round(attack_ratio / (1 - attack_ratio) * len(normal))
    attacks = AttackGenerator(normal, payloads, noise, seed).generate(n_att)
    base = normal.assign(label=0, attack_type="normal", session_id=-1)
    out = pd.concat([base, attacks], ignore_index=True)
    return out.sort_values("ts", ignore_index=True)
