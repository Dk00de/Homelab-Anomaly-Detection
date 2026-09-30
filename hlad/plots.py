"""Figuras y tablas del reporte a partir de results/runs.csv (y dataset_comparison.csv).

Uso: python -m hlad.plots
Salida: results/figures/*.png y results/tables/*.csv (la tabla gemela de cada figura).

Convenciones: cada modelo conserva su color en todas las figuras (paleta
categórica de referencia en su orden validado); magnitudes (recall, matrices de
confusión) en una sola rampa azul; barras de error = desviación estándar entre
semillas.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .paths import RESULTS

FIG = RESULTS / "figures"
TAB = RESULTS / "tables"

# Paleta (referencia dataviz, modo claro)
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MODEL_COLOR = {"arbol": "#2a78d6", "naive_bayes": "#eb6834", "kmeans": "#1baf7a", "fail2ban": "#eda100"}
MODEL_LABEL = {"arbol": "Árbol de decisión", "naive_bayes": "Naive Bayes", "kmeans": "K-means", "fail2ban": "Regla fail2ban"}
MODELS = ["arbol", "naive_bayes", "kmeans"]
BLUES = LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])
SCEN_LABEL = {"A": "A · entrenado con CSIC", "B_contenido": "B · homelab, solo contenido",
              "B_completo": "B · homelab, contenido + contexto"}
NOISES = ["bajo", "medio", "alto"]
FAMILIES = ["inyeccion", "escaneo", "fuerza_bruta"]
FAMILY_LABEL = {"inyeccion": "Inyección", "escaneo": "Escaneo", "fuerza_bruta": "Fuerza bruta"}

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
    "axes.titlesize": 11, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "legend.labelcolor": INK2,
})


def _save(fig, name: str, table: pd.DataFrame) -> None:
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    table.to_csv(TAB / f"{name}.csv")
    print(f"  {name}.png  +  tables/{name}.csv")


def _base(runs: pd.DataFrame) -> pd.DataFrame:
    return runs[~runs["escenario"].str.endswith("+fail2ban")]


def _line(ax, x, mean, std, model):
    c = MODEL_COLOR[model]
    ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.10, linewidth=0)
    ax.plot(x, mean, color=c, linewidth=2, marker="o", markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=MODEL_LABEL[model])
    return mean[-1]


def _end_labels(ax, x_end, values, min_gap=0.05):
    """Etiqueta el valor final de cada línea, separando las que quedarían encimadas."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = []
    for i in order:
        y = values[i] if not placed else max(values[i], placed[-1] + min_gap)
        placed.append(y)
        ax.annotate(f"{values[i]:.2f}", (x_end, y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)


def _fig_legend(fig, models):
    handles = [plt.Line2D([], [], color=MODEL_COLOR[m], linewidth=2, marker="o", markersize=6,
                          markeredgecolor=SURFACE) for m in models]
    fig.legend(handles, [MODEL_LABEL[m] for m in models], loc="upper right", ncol=len(models),
               fontsize=9, bbox_to_anchor=(1.0, 1.0))


def fig_ruido(runs):
    """F1 por nivel de ruido, un panel por escenario (10% de ataques)."""
    d = _base(runs)
    d = d[(d["ataques"] == 0.1) & d["escenario"].isin(SCEN_LABEL)]
    g = d.groupby(["escenario", "modelo", "ruido"])["f1"].agg(["mean", "std"]).reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    x = np.arange(3)
    for ax, scen in zip(axes, SCEN_LABEL):
        ends = []
        for m in MODELS:
            s = g[(g["escenario"] == scen) & (g["modelo"] == m)].set_index("ruido").reindex(NOISES)
            ends.append(_line(ax, x, s["mean"].to_numpy(), s["std"].to_numpy(), m))
        _end_labels(ax, x[-1], ends)
        ax.set_title(SCEN_LABEL[scen], loc="left")
        ax.set_xticks(x, ["Bajo", "Medio", "Alto"])
        ax.set_xlabel("Nivel de ruido (camuflaje)")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(-0.2, 2.45)
    axes[0].set_ylabel("F1")
    fig.suptitle("F1 según el ruido — 10% de ataques, media ± desv. estándar (5 semillas)",
                 x=0.01, y=1.0, ha="left", va="top", color=INK, fontsize=12, fontweight="bold")
    _fig_legend(fig, MODELS)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    table = g.pivot_table(index=["escenario", "modelo"], columns="ruido", values=["mean", "std"]).round(3)
    _save(fig, "01_f1_por_ruido", table)


def fig_desbalance(runs):
    """F1 según la proporción de ataques (B_completo, ruido medio) + regla fail2ban."""
    d = _base(runs)
    d = d[(d["ruido"] == "medio") & (d["escenario"].isin(["B_completo", "regla"]))]
    g = d.groupby(["modelo", "ataques"])["f1"].agg(["mean", "std"]).reset_index()
    ratios = [0.5, 0.1, 0.01]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ends = []
    for m in MODELS + ["fail2ban"]:
        s = g[g["modelo"] == m].set_index("ataques").reindex(ratios)
        ends.append(_line(ax, x, s["mean"].to_numpy(), s["std"].to_numpy(), m))
    _end_labels(ax, x[-1], ends)
    ax.set_xticks(x, ["50%", "10%", "1%"])
    ax.set_xlabel("Proporción de ataques en el tráfico  →  más realista")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.2, 2.35)
    ax.set_title("A menos ataques, peor F1 — B_completo, ruido medio", loc="left")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    _save(fig, "02_f1_por_desbalance", g.pivot_table(index="modelo", columns="ataques",
                                                     values=["mean", "std"]).round(3))


def fig_falsos_positivos(runs):
    """FPR sobre tráfico legítimo: A vs B (ruido medio, 10%)."""
    d = _base(runs)
    d = d[(d["ruido"] == "medio") & (d["ataques"] == 0.1) & d["escenario"].isin(["A", "B_completo"])]
    g = d.groupby(["escenario", "modelo"])["fpr"].agg(["mean", "std"]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, scen in zip(axes, ["A", "B_completo"]):
        s = g[g["escenario"] == scen].set_index("modelo").reindex(MODELS)
        y = np.arange(len(MODELS))
        ax.barh(y, s["mean"] * 100, height=0.55, color=[MODEL_COLOR[m] for m in MODELS],
                xerr=s["std"] * 100, error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 0})
        for yi, v in zip(y, s["mean"] * 100):
            ax.annotate(f"{v:.1f}%", (v, yi), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        ax.set_yticks(y, [MODEL_LABEL[m] for m in MODELS])
        ax.set_xlim(0, 105)
        ax.set_xlabel("Peticiones legítimas marcadas como ataque (%)")
        ax.set_title(SCEN_LABEL[scen], loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()  # eje compartido: invertir una sola vez
    fig.suptitle("Falsos positivos sobre tráfico legítimo real — ruido medio, 10% de ataques",
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "03_falsos_positivos_A_vs_B", g.pivot_table(index="modelo", columns="escenario",
                                                           values=["mean", "std"]).round(4))


def _heat(ax, M, row_labels, col_labels, fmt, vmax=1.0):
    ax.imshow(M, cmap=BLUES, vmin=0, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=10,
                    color="#ffffff" if v / vmax > 0.55 else INK)
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)


def fig_familias(runs):
    """Recall por familia de ataque: modelo, regla y combinación (ruido medio, 10%)."""
    d = runs[(runs["ruido"] == "medio") & (runs["ataques"] == 0.1)]
    rows = [
        ("B_contenido", "arbol", "Árbol (contenido)"),
        ("B_contenido", "naive_bayes", "Naive Bayes (contenido)"),
        ("B_contenido", "kmeans", "K-means (contenido)"),
        ("B_completo", "arbol", "Árbol (contenido + contexto)"),
        ("regla", "fail2ban", "Regla fail2ban sola"),
        ("B_contenido+fail2ban", "arbol", "Árbol (contenido) + fail2ban"),
        ("B_completo+fail2ban", "arbol", "Árbol (completo) + fail2ban"),
    ]
    cols = [f"recall_{f}" for f in FAMILIES] + ["recall"]
    M = np.array([d[(d["escenario"] == s) & (d["modelo"] == m)][cols].mean().to_numpy() for s, m, _ in rows])
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    _heat(ax, M, [r[2] for r in rows], [FAMILY_LABEL[f] for f in FAMILIES] + ["Total"], lambda v: f"{v:.2f}")
    ax.xaxis.tick_top()
    ax.axhline(4.5, color=SURFACE, linewidth=3)
    ax.axvline(2.5, color=SURFACE, linewidth=3)
    ax.set_title("Recall por familia de ataque — ruido medio, 10% de ataques", loc="left", pad=28)
    fig.tight_layout()
    _save(fig, "04_recall_por_familia",
          pd.DataFrame(M, index=[r[2] for r in rows], columns=cols).round(3))


def fig_confusion(runs):
    """Matrices de confusión sumadas sobre las 5 semillas: árbol A vs árbol B_completo."""
    d = runs[(runs["ruido"] == "medio") & (runs["ataques"] == 0.1) & (runs["modelo"] == "arbol")]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    table = {}
    for ax, scen in zip(axes, ["A", "B_completo"]):
        s = d[d["escenario"] == scen][["tn", "fp", "fn", "tp"]].sum()
        M = np.array([[s["tn"], s["fp"]], [s["fn"], s["tp"]]], dtype=float)
        pct = M / M.sum(axis=1, keepdims=True)  # color = % de la clase real
        ax.imshow(pct, cmap=BLUES, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(M[i, j]):,}\n{pct[i, j]:.1%}", ha="center", va="center", fontsize=10,
                        color="#ffffff" if pct[i, j] > 0.55 else INK)
        ax.set_xticks([0, 1], ["Normal", "Ataque"])
        ax.set_yticks([0, 1], ["Normal", "Ataque"])
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.grid(False)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"Árbol · {SCEN_LABEL[scen]}", loc="left", fontsize=10)
        table[scen] = s
    fig.suptitle("Matriz de confusión — ruido medio, 10% de ataques, 5 semillas sumadas",
                 x=0.01, ha="left", color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "05_matriz_confusion_arbol", pd.DataFrame(table))


def fig_dominio():
    """Desplazamiento de dominio: KS por variable, CSIC normal vs homelab normal."""
    path = RESULTS / "dataset_comparison.csv"
    if not path.exists():
        print("  (sin dataset_comparison.csv: corre python -m hlad.compare_datasets)")
        return
    c = pd.read_csv(path, index_col=0)["ks_csicN_vs_home"].sort_values(ascending=False).head(12)
    fig, ax = plt.subplots(figsize=(6.4, 4))
    y = np.arange(len(c))
    ax.barh(y, c.to_numpy(), height=0.6, color=INK2)
    for yi, v in zip(y, c.to_numpy()):
        ax.annotate(f"{v:.2f}", (v, yi), xytext=(6, 0), textcoords="offset points", va="center",
                    fontsize=9, color=INK)
    ax.set_yticks(y, c.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Estadístico KS (0 = misma distribución, 1 = disjuntas)")
    ax.set_title("Tráfico normal: CSIC 2010 vs homelab — 12 variables más distintas", loc="left")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "06_desplazamiento_dominio", c.to_frame("ks").round(3))


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(RESULTS / "runs.csv")
    fig_ruido(runs)
    fig_desbalance(runs)
    fig_falsos_positivos(runs)
    fig_familias(runs)
    fig_confusion(runs)
    fig_dominio()


if __name__ == "__main__":
    main()
