"""Los tres modelos comparados, con una interfaz común: fit(X, y) y score(X) → riesgo.

- Árbol de decisión y Naive Bayes: supervisados, aprenden de ataques etiquetados.
- K-means: detección de novedades. Solo ve tráfico normal (ignora los ataques del
  entrenamiento) y el riesgo es la distancia al centroide más cercano. No necesita
  etiquetas de ataque, solo una línea base limpia.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier


def _signed_log(X):
    # Los conteos (longitudes, caracteres especiales) tienen colas muy largas;
    # log los acerca a una normal (supuesto de GaussianNB) y evita que dominen
    # las distancias de K-means. El signo conserva hour_sin / hour_cos.
    return np.sign(X) * np.log1p(np.abs(X))


class TreeModel:
    name = "arbol"

    def __init__(self, seed: int):
        self.clf = DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, random_state=seed)

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def score(self, X):
        return self.clf.predict_proba(X)[:, 1]

    threshold = 0.5


class NaiveBayesModel:
    name = "naive_bayes"

    def __init__(self, seed: int):
        self.clf = make_pipeline(FunctionTransformer(_signed_log), GaussianNB())

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def score(self, X):
        return self.clf.predict_proba(X)[:, 1]

    threshold = 0.5


class FlooredScaler(StandardScaler):
    """StandardScaler con un piso para la desviación estándar.

    Si una variable es constante en el tráfico normal (p. ej. tok_sensitive = 0 siempre),
    StandardScaler usa escala 1 por convención y una ruta como /.env apenas mueve la
    distancia. Con el piso, cualquier desviación de algo que nunca cambia pesa mucho.
    """

    def __init__(self, floor: float = 0.1, *, copy=True, with_mean=True, with_std=True):
        super().__init__(copy=copy, with_mean=with_mean, with_std=with_std)
        self.floor = floor

    def fit(self, X, y=None, sample_weight=None):
        super().fit(X, y, sample_weight)
        self.scale_ = np.maximum(np.sqrt(self.var_), self.floor)
        return self


class KMeansModel:
    name = "kmeans"

    def __init__(self, seed: int, k: int = 8, normal_quantile: float = 0.99):
        self.prep = make_pipeline(FunctionTransformer(_signed_log), FlooredScaler())
        self.km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        self.q = normal_quantile

    def fit(self, X, y):
        Xn = self.prep.fit_transform(np.asarray(X)[np.asarray(y) == 0])
        self.km.fit(Xn)
        # Umbral: percentil q de las distancias del propio tráfico normal
        # → ~1% de falsos positivos si la prueba se parece al entrenamiento.
        self.threshold = float(np.quantile(self.km.transform(Xn).min(axis=1), self.q))
        return self

    def score(self, X):
        return self.km.transform(self.prep.transform(np.asarray(X))).min(axis=1)


MODELS = [TreeModel, NaiveBayesModel, KMeansModel]
