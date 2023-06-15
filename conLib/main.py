import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from entrenamiento import Entrenamiento
from matplotlib.colors import ListedColormap


def main():
    entrenamiento = Entrenamiento(0.001, 100)
    datos = pd.read_csv('iris.csv', header=None)

    # Procesar los datos, excluir Iris-virginica (datos 101 al 150).
    # Extraccion de los datos, del 0 al 100, los primeros 4 valores de cada fila.
    etiquetas = datos.iloc[0:100, 4].values
    etiquetas = np.where(etiquetas == 'Iris-setosa', -1, 1)  # Simplificamos, para procesarlo en el perceptron.

    # Muestras de flores.
    mue1 = datos.iloc[0:100, [0, 1, 2]].values  # sepal length, sepal width and petal legth
    mue2 = datos.iloc[0:100, [1, 2, 3]].values  # sepal width, petal legth and petal width
    mue3 = datos.iloc[0:100, [2, 3, 0]].values  # petal legth, petal width and sepal length

    # Bucle de entrenamiento.
    entrenamiento.fitness(mue2, etiquetas)  # Altera las entradas con las muestras extraidas del Iris dataset.
    print("Numero de errores por iteracion / epoca: ", entrenamiento.errores)
    print("Pesos resultantes tras las iteraciones: ", entrenamiento.pesos)

    """
    # Mostrar margen de error. (Dr.Carlos)
    plt.plot(range(1, len(entrenamiento.errores) + 1), entrenamiento.errores, marker='o')
    plt.xlabel('Iteraciones / Epocas')
    plt.ylabel('Numero de Errores')
    plt.title("Resultados")
    plt.show()
    """


if __name__ == '__main__':
    main()
