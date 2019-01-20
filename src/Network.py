import numpy as np

# from src.CsvReader import get_data, normalize_data, seperate_inputs_and_outputs
from CsvReader import *


def get_rating(result):
    good = 0.65
    avg = 0.45

    if result >= good:
        return "good"
    elif result >= avg:
        return "average"
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


# print("CORRECT: %d WRONG: %d  ratio = %f" % (correct, wrong, correct / len(testing_inputs) * 100))
# network = Network()
# network.learningRate = 0.4
#
# for i in range(1, 12):
#     result = network.feed_forward([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]) #expected 5
#     print(result)
#     network.backward_propagation(0.5, result, [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4])

# network = Network()
# network.learningRate = 0.2

# whole_data = get_normalized_data('../data/winequality-red.csv')
# edge_row = 1200
# training_data = get_training_data(edge_row, whole_data)
# testing_data = get_testing_data(edge_row, whole_data)

# training_inputs = []
# training_outputs = []
# testing_inputs = []
# testing_outputs = []

# seperate_inputs_and_outputs(training_data, training_inputs, training_outputs)
# seperate_inputs_and_outputs(testing_data, testing_inputs, testing_outputs)

# n_epoch = 500

# for i in range(0, n_epoch):
#     loss_sum = 0
#     for j in range(0, len(training_data)):
#         result = network.feed_forward(training_inputs[j])
#         network.backward_propagation(training_outputs[j][0], result, training_inputs[j])
#         # print("Epoch= %d, data_row=%f, error=%f, expected=%f" % (i, j, result, training_outputs[j][0]))
#         loss_sum += abs(result - training_outputs[j][0])

#     print("Epoch %d, loss sum = %f" % (i, loss_sum))

# wrong = 0
# correct = 0
# print("\n===TESTING===\n")
# for i in range(0, len(testing_data)):
#     result = network.feed_forward(testing_inputs[i]) \
#         # if round(result, 1) == testing_outputs[i][0]:
#     if get_rating(result) == get_rating(testing_outputs[i][0]):
#         print("ROW %d - CORRECT" % i)
#         print(result, testing_outputs[i][0])
#         correct += 1
#     else:
#         print("ROW %d - WRONG!!! \n %f %f" % (i, result, testing_outputs[i][0]))
#         print(result, testing_outputs[i][0])
#         wrong += 1

# print("CORRECT: %d WRONG: %d  ratio = %f" % (correct, wrong, correct / len(testing_inputs) * 100))


# save_network_to_file(network, "network.txt")
# network2 = load_network_from_file("network.txt")
# print(network2.W1)
# print(network2.W2)
# print(network2.B1)
# print(network2.B2)

