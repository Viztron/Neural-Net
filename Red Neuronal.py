from numpy import exp, array, random, dot


class NeuralNetwork():
    def __init__(self):
        # Generamos númneros aleatorios

        random.seed(1)

        # Modelamos la neurona, con 3 conexiones de entrada y 1 de salida.
        # Asignamos pesos aleatorios a la matriz 3x1, con valores de -1 a 1
        # and mean 0.
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    # La función Sigmoide
    # Hacemos pasar los pesos por ella, para normalizarlos y obtener valores entre 0 y 1

    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # La derivada de la Sigmoide, es decir, el gradiente descendente
    # Esto nos indica lo segura que es la función
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # Ahora entrenamos la neurona
    # En cada repetición ajustamos los pesos sinápticos
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pasamos los datos de un ejemplo por nuestra neurona
            output = self.think(training_set_inputs)

            # Cálculo del error: la diferencia entre el valor obtenido y el deseado
            error = training_set_outputs - output

            # Multiplicamos el error por el valor de entrada y esto, a su vez, por el gradiente descendente
            # Este proceso ajustará a los pesos seguros
            # Los datos de entrada que sean 0 no será necesario ajustarlos pues al multiplicarlos anularán al resto de factores
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

            # Ajuste de los pesos
            self.synaptic_weights += adjustment

    # Ahora hacemos pensar a la neurona
    def think(self, inputs):
    #Hacemos pasar los datos de entrada a través de nuestra neurona
        return self.__sigmoid(dot(inputs, self.synaptic_weights))


if __name__ == "__main__":

    #Iniciamos el funcionamiento de la neurona
    neural_network = NeuralNetwork()

    print ("Pesos iniciales aleatorios: ")
    print (neural_network.synaptic_weights)

    # El set de datos de entrada de los 4 ejemplos
    # y el de datos de salida
    training_set_inputs = array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = array([[0, 1, 1, 0]]).T

    # Con estos datos, entrenamos a la neurona
    # Repetimos el proceso , haciendo pequeños ajustes por cada vez
    neural_network.train(training_set_inputs, training_set_outputs, 10000)

    print ("Nuevos pesos despues del entrenamiento: ")
    print (neural_network.synaptic_weights)

    # Testeamos la neurona para que resuelva una nueva situación. En este caso el resultado debería ser 1, pues el primer númnero
	#empezando por la izquierda es un 1
    print ("Considerando la nueva situacion: [1, 0, 0] -> ?: ")
    print (neural_network.think(array([1, 0, 0])))
