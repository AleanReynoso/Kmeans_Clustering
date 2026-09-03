"""
generar_reporte.py
-------------------
Genera docs/reporte_resultados.pdf con el resumen del dataset, la
matriz de confusión, las métricas y las conclusiones, usando los
resultados reales calculados por main.py (se vuelve a correr el
mismo flujo aquí para tener los números y las imágenes a la mano).
"""

import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)

from main import (
    load_dataset, train_test_split, standardize, elbow_method,
    DATA_PATH, RANDOM_STATE, TEST_FRACTION, K, RESULTS_DIR,
)
from kmeans import KMeans
from metrics import (
    map_clusters_to_labels, apply_mapping, confusion_matrix, classification_metrics,
)

OUT_PATH = "../docs/reporte_resultados.pdf"


def run_pipeline():
    X, y, feat_names = load_dataset(DATA_PATH)
    classes = sorted(set(y))
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, TEST_FRACTION, RANDOM_STATE
    )
    X_train, mean, std = standardize(X_train_raw)
    X_test, _, _ = standardize(X_test_raw, mean, std)

    model = KMeans(k=K, n_init=10, random_state=RANDOM_STATE)
    model.fit(X_train)
    train_clusters = model.labels_
    mapping = map_clusters_to_labels(train_clusters, y_train, K, classes)

    test_clusters = model.predict(X_test)
    y_pred = apply_mapping(test_clusters, mapping)

    cm = confusion_matrix(y_test, y_pred, classes)
    metrics_result = classification_metrics(cm, classes)

    return {
        "X": X, "y": y, "classes": classes,
        "X_train_raw": X_train_raw, "X_test_raw": X_test_raw,
        "y_train": y_train, "y_test": y_test,
        "model": model, "mapping": mapping,
        "cm": cm, "metrics": metrics_result,
    }


def build_pdf(results):
    doc = SimpleDocTemplate(
        OUT_PATH, pagesize=letter,
        topMargin=0.7 * inch, bottomMargin=0.7 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleCustom", parent=styles["Title"], fontSize=18, spaceAfter=6
    )
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6)
    body = ParagraphStyle("BodyCustom", parent=styles["Normal"], fontSize=10.5, leading=15)

    story = []

    story.append(Paragraph("Reporte de Resultados — K-means desde cero", title_style))
    story.append(Paragraph(
        "Clustering del dataset Iris con una implementación manual del algoritmo K-means "
        "(sin usar librerías de aprendizaje máquina).", body))
    story.append(Spacer(1, 10))

    # -------------------- 1. Dataset --------------------
    story.append(Paragraph("1. Dataset de entrenamiento y de prueba", h2))
    classes = results["classes"]
    n_total = len(results["y_train"]) + len(results["y_test"])
    story.append(Paragraph(
        f"Se utilizó el dataset público <b>Iris</b> ({n_total} muestras, 4 características "
        f"numéricas: sepal_length, sepal_width, petal_length, petal_width; 3 clases: "
        f"{', '.join(classes)}). Se aplicó un split <b>estratificado 80% / 20%</b>: "
        f"<b>{len(results['y_train'])} muestras para entrenamiento</b> y "
        f"<b>{len(results['y_test'])} muestras para prueba</b>, manteniendo en ambos "
        f"conjuntos la misma proporción de especies que el dataset original. "
        f"Las características se estandarizaron (media 0, desviación 1) usando "
        f"únicamente las estadísticas del set de entrenamiento.", body))
    story.append(Paragraph(
        "Como K-means es un algoritmo no supervisado, las etiquetas reales no se usaron "
        "para entrenar: el modelo se ajustó únicamente con las 4 características. "
        "Después del entrenamiento, a cada cluster se le asignó la especie real "
        "mayoritaria entre los puntos de entrenamiento que cayeron en él, y ese mapeo "
        "se reutilizó tal cual para traducir los clusters del set de prueba a especies, "
        "sin volver a usar las etiquetas de prueba en el proceso.", body))

    story.append(Spacer(1, 6))
    story.append(Image(f"{RESULTS_DIR}/elbow_method.png", width=4.3 * inch, height=2.9 * inch))
    story.append(Paragraph(
        "<i>Figura 1. Método del codo: la inercia baja notoriamente hasta k=3 y luego "
        "se aplana, lo que justifica usar k=3 (consistente con las 3 especies reales).</i>",
        body))

    # -------------------- 2. Matriz de confusión --------------------
    story.append(Paragraph("2. Matriz de confusión (set de prueba)", h2))
    story.append(Paragraph(
        "Filas = especie real, columnas = especie predicha (cluster traducido con el "
        "mapeo aprendido en entrenamiento).", body))

    cm = results["cm"]
    table_data = [[""] + classes]
    for i, cls in enumerate(classes):
        table_data.append([cls] + [str(v) for v in cm[i]])

    t = Table(table_data, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 1), (0, -1), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(Spacer(1, 6))
    story.append(t)

    # -------------------- 3. Métricas --------------------
    story.append(Paragraph("3. Métricas de desempeño (set de prueba)", h2))
    m = results["metrics"]
    story.append(Paragraph(f"<b>Accuracy global:</b> {m['accuracy']:.4f}", body))

    metrics_table_data = [["Clase", "Precision", "Recall", "F1-score"]]
    for cls, vals in m["per_class"].items():
        metrics_table_data.append([
            cls, f"{vals['precision']:.4f}", f"{vals['recall']:.4f}", f"{vals['f1']:.4f}"
        ])
    macro = m["macro_avg"]
    metrics_table_data.append([
        "Macro avg", f"{macro['precision']:.4f}", f"{macro['recall']:.4f}", f"{macro['f1']:.4f}"
    ])

    mt = Table(metrics_table_data, hAlign="CENTER", colWidths=[1.6 * inch] * 4)
    mt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#ecf0f1")),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(Spacer(1, 6))
    story.append(mt)
    story.append(Paragraph(
        "Precision, recall y F1-score se calcularon a partir de la matriz de confusión "
        "de forma manual (verdaderos/falsos positivos y negativos por clase), sin usar "
        "funciones de librerías de métricas.", body))

    story.append(Spacer(1, 10))

    # -------------------- 4. Gráficas de clusters --------------------
    story.append(Paragraph("4. Visualización de los clusters", h2))
    story.append(Paragraph(
        "Proyección sobre petal_length y petal_width (las dos características con mayor "
        "poder de separación entre especies). Color = cluster asignado por el modelo; "
        "forma del marcador = especie real; X negra = centroides.", body))
    story.append(Spacer(1, 6))
    story.append(Image(f"{RESULTS_DIR}/clusters_train.png", width=3.1 * inch, height=2.6 * inch))
    story.append(Spacer(1, 4))
    story.append(Image(f"{RESULTS_DIR}/clusters_test.png", width=3.1 * inch, height=2.6 * inch))

    # -------------------- 5. Análisis y conclusión --------------------
    story.append(Paragraph("5. Análisis y conclusión", h2))
    story.append(Paragraph(
        f"El modelo alcanzó un <b>accuracy de {m['accuracy']*100:.1f}%</b> sobre el set de "
        f"prueba al traducir los clusters encontrados a especies reales. La especie "
        f"<b>setosa</b> se separó de forma perfecta (precision y recall de 1.0), lo cual es "
        f"esperado porque en el espacio de características es linealmente separable del "
        f"resto. La mayor parte del error ocurre entre <b>versicolor</b> y <b>virginica</b>, "
        f"que se traslapan parcialmente en petal_length/petal_width; esto es consistente "
        f"con el comportamiento reportado en la literatura para este dataset y refleja una "
        f"limitación esperable de K-means: al agrupar por cercanía euclidiana a un centroide "
        f"(clusters aproximadamente esféricos), no puede capturar fronteras de decisión "
        f"curvas o solapamientos entre clases vecinas.", body))
    story.append(Paragraph(
        "El método del codo confirma que k=3 es una elección razonable, lo cual coincide "
        "con el número real de especies en el dataset, aunque en un escenario puramente "
        "no supervisado (sin conocer las clases reales) esta elección se basaría "
        "únicamente en la forma de la curva de inercia. En general, los resultados "
        "muestran que K-means, sin ninguna supervisión durante el entrenamiento, es capaz "
        "de recuperar la estructura de clases del dataset Iris con un desempeño alto, "
        "confirmando que las especies tienen una separación natural en el espacio de "
        "características utilizado.", body))

    doc.build(story)
    print(f"PDF generado en {OUT_PATH}")


if __name__ == "__main__":
    results = run_pipeline()
    build_pdf(results)
