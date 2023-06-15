import numpy as np


class Entrenamiento(object):
    def __init__(self, factorA=0.001, numIter=10):
        self.factorA = factorA  # Factor de aprendizaje
        self.numIter = numIter  # Epocas

    def fitness(self, X, y):
        self.pesos = np.zeros(1 + X.shape[1])
        self.errores = []
        for i in range(self.numIter):
            error = 0
            for j, xObjetivo in zip(X, y):
                deltaW = self.factorA * (xObjetivo - self.predecir(j))
                self.pesos[1:] += deltaW * j
                self.pesos[0] += deltaW
                error += int(deltaW != 0.0)
            self.errores.append(error)
        return self

    def entradas(self, X):
        return np.dot(X, self.pesos[1:]) + self.pesos[0]

    def predecir(self, X):
        return np.where(self.entradas(X) >= 0.0, 1, -1)
