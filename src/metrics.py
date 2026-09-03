"""
metrics.py
----------
Métricas de evaluación implementadas manualmente (sin sklearn.metrics
ni ninguna otra librería de estadística/ML avanzada): matriz de
confusión, accuracy, precision, recall y F1-score (macro-promedio).

Como K-means es un algoritmo no supervisado, no genera "clase 0",
"clase 1", etc. con el mismo significado que las etiquetas reales.
Por eso, antes de construir la matriz de confusión, se hace un
mapeo cluster -> clase real usando la clase mayoritaria dentro de
cada cluster (esto se calcula sobre el set de entrenamiento y
después se reutiliza tal cual sobre el set de prueba).
"""

import numpy as np


def map_clusters_to_labels(cluster_labels, true_labels, n_clusters, classes):
    """
    Para cada cluster, encuentra la clase real más frecuente entre los
    puntos que cayeron en él y construye un diccionario
    {cluster_id: clase_real}.
    """
    mapping = {}
    for c in range(n_clusters):
        mask = cluster_labels == c
        if mask.sum() == 0:
            mapping[c] = classes[0]
            continue
        true_in_cluster = true_labels[mask]
        # conteo manual de frecuencias
        counts = {cls: 0 for cls in classes}
        for lbl in true_in_cluster:
            counts[lbl] += 1
        mapping[c] = max(counts, key=counts.get)
    return mapping


def apply_mapping(cluster_labels, mapping):
    """Traduce cada etiqueta de cluster a la clase real correspondiente
    según el mapeo dado."""
    return np.array([mapping[c] for c in cluster_labels])


def confusion_matrix(y_true, y_pred, classes):
    """
    Construye la matriz de confusión manualmente.
    Filas = clase real, Columnas = clase predicha.
    """
    n = len(classes)
    index = {cls: i for i, cls in enumerate(classes)}
    matrix = np.zeros((n, n), dtype=int)
    for true_val, pred_val in zip(y_true, y_pred):
        i = index[true_val]
        j = index[pred_val]
        matrix[i, j] += 1
    return matrix


def classification_metrics(cm, classes):
    """
    A partir de la matriz de confusión calcula, por clase:
    precision, recall y F1-score; además del accuracy global y los
    promedios macro.
    """
    n = len(classes)
    total = cm.sum()
    accuracy = np.trace(cm) / total if total > 0 else 0.0

    per_class = {}
    precisions, recalls, f1s = [], [], []

    for i, cls in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        per_class[cls] = {"precision": precision, "recall": recall, "f1": f1}
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    macro = {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
    }

    return {
        "accuracy": float(accuracy),
        "per_class": per_class,
        "macro_avg": macro,
    }
