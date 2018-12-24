import numpy as np

# from src.CsvReader import get_data, normalize_data, seperate_inputs_and_outputs
from src.CsvReader import *


class Network:

    def __init__(self, inputSize=11, hiddenSize=4):
        self.outputSize = 1
        self.learningRate = 0.05
        self.input_size = inputSize
        self.hidden_size = hiddenSize
        self.W1 = np.random.randn(inputSize, hiddenSize) / np.sqrt(inputSize)
        self.W2 = np.random.randn(hiddenSize, self.outputSize) / np.sqrt(hiddenSize)
        self.B1 = np.zeros((1, hiddenSize))
        self.B2 = np.zeros((1, self.outputSize))
        self.A1 = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feed_forward(self, input):
        Z1 = (np.asmatrix(np.dot(input, self.W1)) + np.asmatrix(self.B1)).getA()
        self.A1 = self.sigmoid(Z1)
        Z2 = (np.asmatrix(np.dot(self.A1, self.W2)) + np.asmatrix(self.B2)).getA()
        output = self.sigmoid(Z2)

        return output

    def backward_propagation(self, expected, calculated, input):
        output_loss = (expected - calculated) * self.sigmoid_derivative(calculated)

        hidden_loss = []
        for i in range(0, self.hidden_size):
            hidden_loss.append(output_loss*self.W2[i, 0] * self.sigmoid_derivative(self.A1[0, i]))

        update_W1 = [[0.0 for x in range(self.hidden_size)] for y in range(self.input_size)]
        for i in range(0, self.input_size):
            for j in range(0, self.hidden_size):
                update_W1[i][j] = hidden_loss[j] * input[i]

        self.W1 = (np.asmatrix(self.W1) + np.asmatrix(update_W1) * self.learningRate).getA()

        update_B1 = []
        for i in range(0, self.hidden_size):
            update_B1.append(hidden_loss[i])

        self.B1 = (np.asmatrix(self.B1) + np.asmatrix(update_B1) * self.learningRate).getA()

        update_W2 = [[0] for i in range(self.hidden_size)]
        for i in range(0, self.hidden_size):
            update_W2[i][0] = output_loss * self.A1[0][i]

        self.W2 = (np.asmatrix(self.W2) + np.asmatrix(update_W2) * self.learningRate).getA()

        update_B2 = np.matrix([
            [output_loss]
        ])

        self.B2 = (np.asmatrix(self.B2) + update_B2 * self.learningRate).getA()


network = Network()
network.learningRate = 0.4

for i in range(1, 12):
    result = network.feed_forward([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]) #expected 5
    print(result)
    network.backward_propagation(0.5, result[0][0], [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4])
