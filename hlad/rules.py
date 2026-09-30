"""Regla estilo fail2ban: la capa de comportamiento que el modelo por petición no ve.

Simula un jail sobre los logs de NPM con los parámetros del jail.local del homelab:
si un origen acumula `maxretry` respuestas 401/403 dentro de `findtime`, todas sus
peticiones durante `bantime` se marcan.

Supuesto de simulación (hoy NO se cumple, ver data/README.md): NPM ve la IP real.
- Cada sesión de ataque sintética = una IP atacante distinta.
- El tráfico normal se agrupa por User-Agent como aproximación de "un dispositivo".
Con la configuración actual todo llega como el gateway de Docker, y un jail así
banearía a todos los usuarios a la vez.
"""
import numpy as np
import pandas as pd

FAIL_STATUS = (401, 403)


def origins(df: pd.DataFrame) -> pd.Series:
    atk = "atk_" + df["session_id"].astype(str)
    norm = "ua_" + df["user_agent"].astype(str)
    return pd.Series(np.where(df["label"] == 1, atk, norm), index=df.index)


def fail2ban_flags(df: pd.DataFrame, maxretry: int = 5, findtime: str = "10min",
                   bantime: str = "10min") -> np.ndarray:
    """Devuelve un arreglo booleano alineado con df: True = petición bloqueada/marcada."""
    win = pd.Timedelta(findtime).value
    ban = pd.Timedelta(bantime).value
    d = pd.DataFrame({
        "origin": origins(df),
        # as_unit("ns"): pandas puede inferir resolución en µs; las ventanas están en ns.
        "t": pd.to_datetime(df["ts"], utc=True).dt.as_unit("ns").astype("int64"),
        "fail": df["status"].isin(FAIL_STATUS).astype(int),
    }, index=df.index)

    flags = pd.Series(False, index=df.index)
    for _, g in d.sort_values("t").groupby("origin", sort=False):
        if g["fail"].sum() < maxretry:
            continue
        t = g["t"].to_numpy()
        fails_cum = g["fail"].cumsum().to_numpy()
        start = np.searchsorted(t, t - win, side="left")
        in_window = fails_cum - np.where(start > 0, fails_cum[start - 1], 0)
        triggers = t[in_window >= maxretry]
        # Marcada si hubo un disparo en los últimos `bantime` (incluye el propio disparo).
        last = np.searchsorted(triggers, t, side="right") - 1
        banned = (last >= 0) & (t - triggers[np.maximum(last, 0)] <= ban)
        flags.loc[g.index] = banned
    return flags.to_numpy()
