"""
main.py
-------
Script principal. Corre por separado únicamente con un intérprete de
Python (no requiere notebook ni IDE):

    python3 main.py

Flujo:
1. Carga el dataset Iris (data/iris.csv).
2. Separa 80% para entrenamiento y 20% para prueba (split manual).
3. Entrena K-means (implementado a mano en kmeans.py) con k=3 sobre
   el set de entrenamiento.
4. Mapea cada cluster a la especie real mayoritaria (usando SOLO el
   set de entrenamiento).
5. Predice los clusters del set de prueba y los traduce a especies
   usando el mapeo aprendido.
6. Calcula matriz de confusión y métricas (accuracy, precision,
   recall, F1) sobre el set de prueba.
7. Genera gráficas (método del codo y clusters en 2D) en results/.
8. Imprime un resumen completo en consola.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # para poder correr sin entorno gráfico
import matplotlib.pyplot as plt

from kmeans import KMeans
from metrics import (
    map_clusters_to_labels,
    apply_mapping,
    confusion_matrix,
    classification_metrics,
)

DATA_PATH = "../data/iris.csv"
RESULTS_DIR = "../results"
RANDOM_STATE = 42
TEST_FRACTION = 0.2
K = 3


def load_dataset(path):
    """Carga el CSV manualmente con numpy/csv, sin pandas.read_csv +
    librerías de ML. (Se usa numpy solo para el manejo de arreglos)."""
    import csv
    features = []
    labels = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            *feat, species = row
            features.append([float(v) for v in feat])
            labels.append(species)
    return np.array(features), np.array(labels), header[:-1]


def train_test_split(X, y, test_fraction, seed):
    """Split manual estratificado por clase: separa test_fraction de
    cada especie para que el set de prueba mantenga la misma
    proporción de clases que el original."""
    rng = np.random.RandomState(seed)
    classes = sorted(set(y))

    train_idx, test_idx = [], []
    for cls in classes:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_fraction))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    return (X[train_idx], X[test_idx], y[train_idx], y[test_idx])


def standardize(X, mean=None, std=None):
    """Estandariza features (media 0, desviación 1). Si no se dan
    mean/std, se calculan sobre X (deben calcularse SOLO con el set
    de entrenamiento y reutilizarse en el de prueba, para no filtrar
    información)."""
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
        std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def elbow_method(X_train, ks, seed):
    inertias = []
    for k in ks:
        model = KMeans(k=k, n_init=5, random_state=seed)
        model.fit(X_train)
        inertias.append(model.inertia_)
    return inertias


def plot_elbow(ks, inertias, out_path):
    plt.figure(figsize=(6, 4))
    plt.plot(list(ks), inertias, marker="o")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Inercia (suma de distancias^2)")
    plt.title("Método del codo")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_clusters(X, true_labels, cluster_labels, centroids, feat_names, out_path,
                   title):
    """Grafica en 2D usando las 2 features con mayor varianza (petal
    length y petal width en el caso de Iris) coloreando por cluster
    predicho, con marcador distinto por especie real."""
    fi, fj = 2, 3  # petal_length, petal_width
    plt.figure(figsize=(6, 5))
    markers = {"setosa": "o", "versicolor": "s", "virginica": "^"}
    colors = plt.cm.tab10(np.linspace(0, 1, len(set(cluster_labels))))

    for cls in sorted(set(true_labels)):
        mask = true_labels == cls
        plt.scatter(
            X[mask, fi], X[mask, fj],
            c=[colors[c] for c in cluster_labels[mask]],
            marker=markers.get(cls, "o"),
            edgecolor="k", linewidth=0.4, s=60,
            label=f"real: {cls}",
        )

    plt.scatter(
        centroids[:, fi], centroids[:, fj],
        c="black", marker="X", s=200, label="centroides",
    )
    plt.xlabel(feat_names[fi])
    plt.ylabel(feat_names[fj])
    plt.title(title)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def print_confusion_matrix(cm, classes):
    header = "        " + "".join(f"{c[:10]:>12}" for c in classes)
    print(header)
    for i, cls in enumerate(classes):
        row = "".join(f"{cm[i, j]:>12}" for j in range(len(classes)))
        print(f"{cls[:8]:>8}{row}")


def main():
    print("=" * 70)
    print("K-MEANS IMPLEMENTADO DESDE CERO - DATASET IRIS")
    print("=" * 70)

    X, y, feat_names = load_dataset(DATA_PATH)
    classes = sorted(set(y))
    print(f"\nDataset cargado: {X.shape[0]} muestras, {X.shape[1]} features")
    print(f"Clases: {classes}")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, TEST_FRACTION, RANDOM_STATE
    )
    print(f"\nEntrenamiento: {X_train_raw.shape[0]} muestras "
          f"({100*(1-TEST_FRACTION):.0f}%)")
    print(f"Prueba:        {X_test_raw.shape[0]} muestras "
          f"({100*TEST_FRACTION:.0f}%)")
    print("(split estratificado: misma proporción de especies en ambos sets)")

    # Estandarización (media/std calculados SOLO con train)
    X_train, mean, std = standardize(X_train_raw)
    X_test, _, _ = standardize(X_test_raw, mean, std)

    # --- Método del codo (para justificar k=3) ---
    ks = range(1, 8)
    inertias = elbow_method(X_train, ks, RANDOM_STATE)
    plot_elbow(ks, inertias, f"{RESULTS_DIR}/elbow_method.png")
    print("\nInercia por valor de k (método del codo):")
    for k, inertia in zip(ks, inertias):
        print(f"  k={k}: inercia = {inertia:.3f}")

    # --- Entrenamiento final con k=3 ---
    model = KMeans(k=K, n_init=10, random_state=RANDOM_STATE)
    model.fit(X_train)
    print(f"\nModelo entrenado con k={K} "
          f"(inercia final = {model.inertia_:.3f}, "
          f"iteraciones = {model.n_iter_})")

    train_clusters = model.labels_
    mapping = map_clusters_to_labels(train_clusters, y_train, K, classes)
    print(f"\nMapeo cluster -> especie (aprendido con el set de entrenamiento):")
    for c, cls in mapping.items():
        print(f"  cluster {c} -> {cls}")

    # --- Predicción sobre el set de prueba ---
    test_clusters = model.predict(X_test)
    y_pred = apply_mapping(test_clusters, mapping)

    # --- Evaluación ---
    cm = confusion_matrix(y_test, y_pred, classes)
    metrics_result = classification_metrics(cm, classes)

    print("\n" + "-" * 70)
    print("MATRIZ DE CONFUSION (set de prueba)")
    print("-" * 70)
    print_confusion_matrix(cm, classes)

    print("\n" + "-" * 70)
    print("METRICAS (set de prueba)")
    print("-" * 70)
    print(f"Accuracy global: {metrics_result['accuracy']:.4f}")
    print(f"\n{'Clase':<12}{'Precision':>12}{'Recall':>12}{'F1-score':>12}")
    for cls, vals in metrics_result["per_class"].items():
        print(f"{cls:<12}{vals['precision']:>12.4f}{vals['recall']:>12.4f}"
              f"{vals['f1']:>12.4f}")
    macro = metrics_result["macro_avg"]
    print(f"{'Macro avg':<12}{macro['precision']:>12.4f}"
          f"{macro['recall']:>12.4f}{macro['f1']:>12.4f}")

    # --- Gráficas de clusters ---
    plot_clusters(
        X_train_raw, y_train, train_clusters, model.centroids_ * std + mean,
        feat_names, f"{RESULTS_DIR}/clusters_train.png",
        "Clusters en entrenamiento",
    )
    plot_clusters(
        X_test_raw, y_test, test_clusters, model.centroids_ * std + mean,
        feat_names, f"{RESULTS_DIR}/clusters_test.png",
        "Clusters en prueba",
    )

    # --- Predicciones de ejemplo en consola ---
    print("\n" + "-" * 70)
    print("EJEMPLOS DE PREDICCION (5 muestras del set de prueba)")
    print("-" * 70)
    for i in range(min(5, len(X_test_raw))):
        print(f"  Features: {X_test_raw[i]} | Real: {y_test[i]:<12} "
              f"| Predicho: {y_pred[i]}")

    print("\nGráficas guardadas en results/: elbow_method.png, "
          "clusters_train.png, clusters_test.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
