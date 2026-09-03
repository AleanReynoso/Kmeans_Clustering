"""
kmeans.py
---------
Implementación manual del algoritmo K-means (clustering no supervisado)
desde cero, sin usar librerías de aprendizaje máquina ni de estadística
avanzada (no sklearn, no scipy.cluster, etc).

Únicamente se usa `numpy` para operaciones aritméticas de arreglos
(sumas, promedios, distancias euclidianas) y `random` para la
inicialización, no para el algoritmo en sí. Toda la lógica de
asignación de clusters, actualización de centroides y criterio de
convergencia está implementada manualmente.

Autor: (agrega tu nombre)
Este archivo se ejecuta directamente con un intérprete de Python:
    python3 kmeans.py
No depende de notebooks ni de ningún IDE.
"""

import random
import numpy as np


class KMeans:
    """
    Implementación manual del algoritmo K-means.

    Parámetros
    ----------
    k : int
        Número de clusters a formar.
    max_iter : int
        Número máximo de iteraciones si no converge antes.
    tol : float
        Tolerancia: si el desplazamiento de los centroides entre una
        iteración y la siguiente es menor a este valor, se considera
        que el algoritmo convergió.
    n_init : int
        Número de veces que se corre el algoritmo completo con
        distintas inicializaciones aleatorias, quedándonos con la
        que produce menor inercia (suma de distancias al cuadrado).
    random_state : int
        Semilla para reproducibilidad.
    """

    def __init__(self, k=3, max_iter=300, tol=1e-4, n_init=10, random_state=42):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.n_init = n_init
        self.random_state = random_state

        self.centroids_ = None
        self.labels_ = None
        self.inertia_ = None
        self.n_iter_ = None

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------
    @staticmethod
    def _euclidean_distance(a, b):
        """Distancia euclidiana entre un punto `a` y un conjunto de
        centroides `b` (calculada manualmente, sin funciones de
        distancia de librerías externas)."""
        diff = a - b
        return np.sqrt(np.sum(diff * diff, axis=-1))

    def _init_centroids(self, X, rng):
        """
        Inicialización tipo K-means++ implementada manualmente:
        1. Se elige el primer centroide al azar entre los puntos.
        2. Cada siguiente centroide se elige con probabilidad
           proporcional al cuadrado de la distancia al centroide más
           cercano ya elegido (esto reduce el riesgo de mínimos
           locales malos comparado con una inicialización totalmente
           aleatoria).
        """
        n_samples = X.shape[0]
        centroids = []

        first_idx = rng.randrange(n_samples)
        centroids.append(X[first_idx])

        for _ in range(1, self.k):
            centroids_arr = np.array(centroids)
            # distancia de cada punto a su centroide más cercano
            dist_sq = np.min(
                np.array([self._euclidean_distance(X[i], centroids_arr) ** 2
                           for i in range(n_samples)]),
                axis=1,
            )
            total = dist_sq.sum()
            if total == 0:
                # todos los puntos coinciden con centroides ya elegidos
                next_idx = rng.randrange(n_samples)
            else:
                probs = dist_sq / total
                cumulative = np.cumsum(probs)
                r = rng.random()
                next_idx = int(np.searchsorted(cumulative, r))
                next_idx = min(next_idx, n_samples - 1)
            centroids.append(X[next_idx])

        return np.array(centroids)

    def _assign_clusters(self, X, centroids):
        """Asigna cada punto al centroide más cercano."""
        n_samples = X.shape[0]
        labels = np.empty(n_samples, dtype=int)
        for i in range(n_samples):
            distances = self._euclidean_distance(X[i], centroids)
            labels[i] = int(np.argmin(distances))
        return labels

    def _update_centroids(self, X, labels, old_centroids):
        """Recalcula cada centroide como el promedio de los puntos
        asignados a él. Si un cluster se queda sin puntos, conserva
        el centroide anterior para evitar errores numéricos."""
        new_centroids = np.empty_like(old_centroids)
        for j in range(self.k):
            points_in_cluster = X[labels == j]
            if len(points_in_cluster) == 0:
                new_centroids[j] = old_centroids[j]
            else:
                new_centroids[j] = points_in_cluster.mean(axis=0)
        return new_centroids

    def _compute_inertia(self, X, labels, centroids):
        """Suma de distancias euclidianas al cuadrado entre cada punto
        y su centroide asignado (métrica interna que K-means intenta
        minimizar)."""
        inertia = 0.0
        for j in range(self.k):
            points_in_cluster = X[labels == j]
            if len(points_in_cluster) > 0:
                diff = points_in_cluster - centroids[j]
                inertia += np.sum(diff * diff)
        return inertia

    # ------------------------------------------------------------------
    # Entrenamiento manual (un solo intento)
    # ------------------------------------------------------------------
    def _fit_once(self, X, rng):
        centroids = self._init_centroids(X, rng)

        for iteration in range(self.max_iter):
            labels = self._assign_clusters(X, centroids)
            new_centroids = self._update_centroids(X, labels, centroids)

            shift = np.sqrt(np.sum((new_centroids - centroids) ** 2, axis=1)).max()
            centroids = new_centroids

            if shift < self.tol:
                break

        labels = self._assign_clusters(X, centroids)
        inertia = self._compute_inertia(X, labels, centroids)
        return centroids, labels, inertia, iteration + 1

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def fit(self, X):
        """
        Ajusta el modelo a los datos X (numpy array de forma
        [n_muestras, n_features]). Corre `n_init` inicializaciones
        distintas y conserva la de menor inercia.
        """
        X = np.asarray(X, dtype=float)
        rng = random.Random(self.random_state)

        best_inertia = None
        best_centroids = None
        best_labels = None
        best_n_iter = None

        for run in range(self.n_init):
            centroids, labels, inertia, n_iter = self._fit_once(X, rng)
            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_n_iter = n_iter

        self.centroids_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def predict(self, X):
        """Asigna cada punto de X al centroide más cercano ya
        aprendido en `fit`."""
        if self.centroids_ is None:
            raise RuntimeError("El modelo no ha sido entrenado. Llama a fit() primero.")
        X = np.asarray(X, dtype=float)
        return self._assign_clusters(X, self.centroids_)

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
