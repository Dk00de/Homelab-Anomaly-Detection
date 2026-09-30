# Homelab-Anomaly-Detection

Proyecto de la materia *Fundamentos de Inteligencia Artificial* (Maestría en IA y Ciberseguridad).

Se comparan tres modelos para detectar peticiones HTTP maliciosas contra los servicios de un homelab:

| Modelo | Tipo | Cómo decide |
|---|---|---|
| K-means | No supervisado | Distancia al centroide más cercano, con un umbral → puntaje de riesgo |
| Árbol de decisión | Supervisado | Reglas aprendidas de ejemplos etiquetados |
| Naive Bayes (gaussiano) | Supervisado, probabilístico | P(anómalo \| variables) |

## Diseño experimental

El homelab solo es accesible por Tailscale y no registró ningún ataque real, así que todo su tráfico (unas 22k peticiones de Nginx Proxy Manager) es de la clase normal. Los ataques de prueba son sintéticos.

| | Experimento A: transferencia | Experimento B: datos propios |
|---|---|---|
| Entrenamiento | [CSIC 2010](data/README.md) | Tráfico normal del homelab + ataques sintéticos |
| Prueba | **Mismo conjunto para A y B:** tráfico normal del homelab (partición de prueba) + ataques sintéticos |
| Variables | Contenido (25) | Contenido + contexto (código de respuesta, bytes, tipo de UA, hora) |
| Pregunta | ¿Sirve un dataset público tal cual? | ¿Qué modelo rinde mejor y cómo le afectan el ruido y el desbalance? |

**Ataques sintéticos** (`hlad/synthetic.py`): inyección (payloads anómalos de CSIC insertados en rutas reales del homelab), escaneo (rutas de reconocimiento conocidas) y fuerza bruta (ráfagas contra endpoints de login reales).

- **Ruido** (bajo / medio / alto): qué tanto se camufla el ataque. Controla el UA de navegador real, el código de respuesta y el horario como los normales, y los payloads sutiles.
- **Desbalance:** proporción de ataques en el dataset.
- Los ataques de CSIC se dividen en tres grupos disjuntos (entrenamiento de A, generación de entrenamiento de B, generación de prueba) para evitar fugas de información.

**Métricas:** precisión, recall, F1, matriz de confusión y PR-AUC. Se prioriza recall.

## Estructura

```
hlad/                  paquete principal
  io_npm.py            parser de access logs de NPM
  io_csic.py           carga de CSIC 2010
  features.py          variables de contenido (ambos datasets) y de contexto (solo homelab)
  synthetic.py         generador de ataques sintéticos (ruido, desbalance)
  models.py            árbol de decisión, Naive Bayes, K-means (detección de novedades)
  rules.py             regla estilo fail2ban (capa de comportamiento)
  experiments.py       escenario × modelo × ruido × desbalance × semilla → results/runs.csv
  plots.py             figuras (results/figures) + tabla gemela de cada una (results/tables)
  build_features.py    data/raw → data/processed
  compare_datasets.py  qué tan distintos son CSIC y el homelab (desplazamiento de dominio)
data/                  no versionado, ver data/README.md
results/               métricas, figuras y tablas generadas
```

## Uso

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m hlad.build_features     # opcional: variables procesadas + medias
python -m hlad.compare_datasets   # desplazamiento de dominio → results/dataset_comparison.csv
python -m hlad.experiments        # ~1 min → results/runs.csv
python -m hlad.plots              # → results/figures/*.png + results/tables/*.csv
```

## Privacidad

Los logs del homelab no se incluyen en el repositorio. Las IPs se anonimizan en el servidor antes de exportarlos (ver `data/README.md`).
