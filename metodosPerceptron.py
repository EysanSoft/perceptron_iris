import random as magico


class Perceptron(object):
    def definirPesos(self, pesos):
        listaPesos = []
        for i in range(len(pesos) - 1):
            listaPesos.append(magico.uniform(0, 1))
        return listaPesos

    def actualizarPesos(self, marErr, datos, pesosActuales):
        listaPesosAct = []
        for i in range(len(pesosActuales)):
            listaPesosAct.append(marErr * float(datos[i]) * magico.uniform(0, 1) + pesosActuales[i])
        return listaPesosAct

    def sumatoria(self, datos, pesosActuales):
        sumaTotal = 0
        for i in range(len(datos) - 1):
            sumaTotal += float(datos[i]) * pesosActuales[i]
        return sumaTotal

    def margenError(self, datos, resAct):
        return int(datos[len(datos) - 1]) - resAct

    def funcionActivacion(self, suma):
        if suma > 0:
            return 1
        else:
            return -1

    def bucleEntrenamiento(self, datos, pesos):
        conAux = 0
        perMet = Perceptron()
        while conAux < len(datos):
            sumatoria = perMet.sumatoria(datos[conAux], pesos)
            funAct = perMet.funcionActivacion(sumatoria)
            marErr = perMet.margenError(datos[conAux], funAct)
            if marErr != 0:
                pesos = perMet.actualizarPesos(marErr, datos[conAux], pesos)
                conAux = 0
            else:
                conAux += 1
        print(pesos)
        return pesos
