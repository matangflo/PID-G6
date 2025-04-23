# Clasificación de Posturas con Machine Learning

Este repositorio contiene un proyecto enfocado en la clasificación de posturas corporales utilizando diferentes algoritmos de machine learning, incluyendo redes neuronales, k-Nearest Neighbors (k-NN) y Random Forest. Se utilizan landmarks extraídos de imágenes como base para los modelos.

## Contenido del Repositorio

- `Redes.ipynb`: Notebook que contiene la creación del dataset `landmarks_dataset.csv` y el modelo de red neuronal con su respectiva prueba. **No es necesario ejecutar la primera Celda ya que es para la creación del .csv y es tardio**
- `landmarks_dataset.csv`: Dataset generado con el dataset descargado desde: 👉 [Yoga Pose Image Classification Dataset](https://www.kaggle.com/datasets/shrutisaxena/yoga-pose-image-classification-dataset)
- `kNN.ipynb`: Implementación del algoritmo k-Nearest Neighbors usando otro csv diferente `knn.csv`.
- `KnnCSV.ipynb`: Creacion del nuevo csv utilizado para el algortimo de KNN.
- `knn.csv`: csv auxiliar utilizado para KNN. **No es necesario ejecutarlo ya que es tardio**
- `RandomForest.ipynb`: Implementación del algoritmo Random Forest utilizando el mismo csv que ha sido utilizado para redes.
- `modelo_pose_mejorado.h5`: Modelo de red neuronal ya entrenado en formato HDF5.
- `scaler.pkl`: Escalador utilizado para normalizar los datos antes de alimentar el modelo.
- `scaler_cnn.pkl`: Escalador específico para el modelo de red neuronal.
- `fotos/`: Carpeta con imágenes de prueba utilizadas para evaluar los modelos.

## Requisitos

- Python 3.8+
- Librerías necesarias (instalables con pip):

```bash
pip install numpy pandas tensorflow scikit-learn matplotlib joblib opencv-python mediapipe
