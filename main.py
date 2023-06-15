from metodosPerceptron import Perceptron


def main():
    # Procesar el dataset.
    archivoDataset = open('iris.csv')
    listaDatos = []
    for i in archivoDataset:
        dato = i.split(',')
        if dato[len(dato) - 1] == "Iris-setosa\n":
            dato[len(dato) - 1] = 1
        elif dato[len(dato) - 1] == "Iris-versicolor\n":
            dato[len(dato) - 1] = -1
        else:
            dato[len(dato) - 1] = "0"
        for j in range(len(dato) - 1):
            dato[j] = float(dato[j])
        if dato[len(dato) - 1] != "0":
            listaDatos.append(dato)
    print(listaDatos)

    # Declarar el perceptron y entrenar los pesos.
    perceptron = Perceptron()
    pesosIniciales = perceptron.definirPesos(listaDatos[0])
    pesosFinales = perceptron.bucleEntrenamiento(listaDatos, pesosIniciales)

    # Evaluar los pesos finales.
    continuar = True
    while continuar:
        mensaje = input("Ingrese (FIN) para terminar, o los valores de una flor para evaluar: ")
        if mensaje == "FIN":
            continuar = False
        else:
            mensaje = mensaje.split(",")
            sumaTotal = perceptron.sumatoria(mensaje, pesosFinales)
            funAct = perceptron.funcionActivacion(sumaTotal)
            if funAct == 1:
                print("Iris-Setosa")
            else:
                print("Iris-Versicolor")


if __name__ == '__main__':
    main()
