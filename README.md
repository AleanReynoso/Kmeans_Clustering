# K-Means desde cero — Clustering del dataset Iris

Implementación manual del algoritmo **K-means** (sin usar `scikit-learn`
ni ninguna otra librería de aprendizaje máquina o estadística avanzada)
aplicada al clásico dataset **Iris** para agrupar flores según sus
medidas de sépalo y pétalo, comparando los clusters obtenidos contra
las especies reales.

## Contenido del repositorio

```
.
├── data/
│   └── iris.csv              # Dataset usado (150 muestras, 3 especies)
├── src/
│   ├── kmeans.py              # Implementación manual de K-means (K-means++ + Lloyd)
│   ├── metrics.py              # Matriz de confusión y métricas implementadas a mano
│   └── main.py                  # Script principal: carga datos, entrena, evalúa, grafica
├── results/
│   ├── elbow_method.png       # Gráfica del método del codo
│   ├── clusters_train.png     # Clusters resultantes en entrenamiento
│   └── clusters_test.png      # Clusters resultantes en prueba
├── docs/
│   └── reporte_resultados.pdf # Reporte con dataset, matriz de confusión, métricas y conclusiones
├── requirements.txt
└── README.md
```

## ¿Qué se implementó desde cero?

- **`kmeans.py`**: la clase `KMeans`, con inicialización tipo K-means++
  (implementada manualmente, no importada), asignación de puntos a
  centroides por distancia euclidiana, actualización de centroides y
  criterio de convergencia, todo escrito a mano. Solo se usa `numpy`
  para operaciones aritméticas de arreglos (no para el algoritmo en
  sí) y `random` para la aleatoriedad de la inicialización.
- **`metrics.py`**: matriz de confusión, accuracy, precision, recall y
  F1-score calculados manualmente (no se usa `sklearn.metrics`).

No se utiliza `scikit-learn`, `scipy.cluster`, ni ninguna otra
librería que ya traiga K-means o métricas de clasificación
implementadas.

## Cómo ejecutar

Requiere solo un intérprete de Python 3 (no depende de ningún IDE ni
notebook):

```bash
# 1. Clonar el repositorio
git clone <URL-del-repo>
cd kmeans-proyecto

# 2. Crear entorno virtual (opcional pero recomendado)
python3 -m venv venv
source venv/bin/activate   # En Windows: venv\Scripts\activate

# 3. Instalar dependencias (numpy, matplotlib: solo para arreglos/gráficas)
pip install -r requirements.txt

# 4. Ejecutar
cd src
python3 main.py
```

El script imprime en consola:
- Tamaño del split de entrenamiento/prueba
- Inercia por valor de k (método del codo)
- El mapeo cluster → especie aprendido
- La matriz de confusión y las métricas sobre el set de prueba
- Ejemplos de predicciones individuales

y guarda las gráficas en `results/`.

## Dataset

Se usó el dataset **Iris** (150 muestras, 4 features numéricas:
`sepal_length`, `sepal_width`, `petal_length`, `petal_width`, y 3
clases: `setosa`, `versicolor`, `virginica`). Se hizo un split
estratificado 80% entrenamiento / 20% prueba, de modo que ambos
conjuntos conservan la misma proporción de especies que el dataset
original.

Como K-means es un algoritmo **no supervisado**, las etiquetas reales
no se usan para entrenar, solo para evaluar: después de entrenar, a
cada cluster se le asigna la especie real mayoritaria dentro de él
(usando solamente el set de entrenamiento), y ese mapeo se reutiliza
tal cual sobre el set de prueba para poder construir la matriz de
confusión.

## Resultados

Ver el reporte completo en [`docs/reporte_resultados.pdf`](docs/reporte_resultados.pdf).


