import numpy as np

# from src.CsvReader import get_data, normalize_data, seperate_inputs_and_outputs
from CsvReader import *


def get_rating(result):
    good = 0.65

    if result >= good:
        return "good"
    else:
        return "bad"

def save_network_to_file(network, filename):
    file = open(filename, "w")
    file.write(str(network.input_size) + "\n")
    file.write(str(network.hidden_size) + "\n")

    for i in range(0, network.input_size):
        for j in range(0, network.hidden_size):
            file.write(str(network.W1[i][j]) + " ")
        file.write("\n")

    for i in range(0, network.hidden_size):
        for j in range(0, network.output_size):
            file.write(str(network.W2[i][j]) + " ")
        file.write("\n")

    for i in range(0, network.hidden_size):
        file.write(str(network.B1[0][i]) + " ")

    file.write("\n")
    file.write(str(network.B2[0][0]) + "\n")

def load_network_from_file(filename):
    try:
        file = open(filename, "r")
    except OSError:
        print("No such file")
        return None

    input_size = int(file.readline())
    hidden_size = int(file.readline())

    W1 = [[float(n) for n in file.readline().split()] for y in range(input_size)]

    W2 = [[float(n) for n in file.readline().split()] for y in range(hidden_size)]

    B1 = [[float(n) for n in file.readline().split()]]

    B2 = [[float(n) for n in file.readline().split()]]

    network = Network(input_size, hidden_size)
    network.W1 = W1
    network.W2 = W2
    network.B1 = B1
    network.B2 = B2

    return network

class Network:

    def __init__(self, inputSize=11, hiddenSize=4):
        self.output_size = 1
        self.learningRate = 0.05
        self.input_size = inputSize
        self.hidden_size = hiddenSize
        self.W1 = np.random.randn(inputSize, hiddenSize) / np.sqrt(inputSize)
        self.W2 = np.random.randn(hiddenSize, self.output_size) / np.sqrt(hiddenSize)
        self.B1 = np.zeros((1, hiddenSize))
        self.B2 = np.zeros((1, self.output_size))
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

        return output[0][0]

    def backward_propagation(self, expected, calculated, input):
        output_loss = (expected - calculated) * self.sigmoid_derivative(calculated)

        hidden_loss = []
        for i in range(0, self.hidden_size):
            hidden_loss.append(output_loss * self.W2[i, 0] * self.sigmoid_derivative(self.A1[0, i]))

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

