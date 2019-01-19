import cupy as cp
import numpy as np

from csvReading import *

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
        #self.learningRate = 0.05
        self.input_size = inputSize
        self.hidden_size = hiddenSize
        self.W1 = cp.random.randn(inputSize, hiddenSize) / cp.sqrt(inputSize)
        self.W2 = cp.random.randn(hiddenSize, self.output_size) / cp.sqrt(hiddenSize)
        self.B1 = cp.zeros((1, hiddenSize))
        self.B2 = cp.zeros((1, self.output_size))
        #self.X1 = cp.zeros((inputSize, hiddenSize))
        #self.X2 = cp.zeros((inputSize, self.output_size))
        self.X1 = cp.zeros((1, hiddenSize))
        self.X2 = cp.zeros((1, self.output_size))
        self.A1 = None

    def sigmoid(self, x):
        return 1 / (1 + cp.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feed_forward(self, input):
        cp.dot(cp.array(input), self.W1, self.X1)
        Z1 = cp.add(self.X1,self.B1)
        self.A1 = self.sigmoid(Z1)

        cp.dot(self.A1, self.W2, self.X2)
        Z2 = cp.add(self.X2, self.B2)
        output = self.sigmoid(Z2)

        return output[0][0]

    def backward_propagation(self, expected, calculated, input):
        output_loss = (expected - calculated) * self.sigmoid_derivative(calculated)

        hidden_loss = []
        for i in range(0, self.hidden_size):
            hidden_loss.append(output_loss * self.W2[i, 0] * self.sigmoid_derivative(self.A1[0, i]))

        #update_W1 = [[0.0 for x in range(self.hidden_size)] for y in range(self.input_size)]
        update_W1 = cp.zeros((self.input_size , self.hidden_size),cp.float32)
        for i in range(0, self.input_size):
            for j in range(0, self.hidden_size):
                update_W1[i][j] = hidden_loss[j] * input[i] * self.learningRate

        self.W1 = cp.add(self.W1, update_W1)

        #update_B1 = []
        update_B1 = cp.zeros(self.hidden_size,cp.float32)

        for i in range(0, self.hidden_size):
            update_B1[i] = hidden_loss[i] * self.learningRate

        self.B1 = cp.add(self.B1, update_B1)

        #update_W2 = [[0] for i in range(self.hidden_size)]
        update_W2 = cp.zeros((self.hidden_size, self.output_size), cp.float32)
        
        for i in range(0, self.hidden_size):
            update_W2[i][0] = output_loss * self.A1[0][i]

        for i in range(0, self.hidden_size):
            for j in range(0, self.output_size):
                update_W2 *= self.learningRate

        self.W2 = cp.add(self.W2,update_W2)

        update_B2 = cp.array(output_loss)

        self.B2 = cp.add(self.B2,(update_B2 * self.learningRate))

        # network = Network()
# network.learningRate = 0.4
#
# for i in range(1, 12):
#     result = network.feed_forward([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]) #expected 5
#     print(result)
#     network.backward_propagation(0.5, result, [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4])

network = Network()
network.learningRate = 0.2

whole_data = get_normalized_data()
edge_row = 1200
training_data = get_training_data(edge_row, whole_data)
testing_data = get_testing_data(edge_row, whole_data)
# training_data = get_specific_data(0, edge_row, whole_data)
# testing_data = get_specific_data(edge_row, len(whole_data), whole_data, "t")

training_outputs = []
testing_outputs = []

seperate_inputs_and_outputs(training_data, training_outputs)
seperate_inputs_and_outputs(testing_data, testing_outputs)

n_epoch = 500

for i in range(0, n_epoch):
     loss_sum = 0
     for j in range(0, len(training_data)):
         result = network.feed_forward(training_data[j])
         network.backward_propagation(training_outputs[j][0], result, training_data[j])
         # print("Epoch= %d, data_row=%f, error=%f, expected=%f" % (i, j, result, training_outputs[j][0]))
         loss_sum += abs(result - training_outputs[j][0])

     print("Epoch %d, loss sum = %f" % (i, loss_sum))

wrong = 0
correct = 0
print("\n===TESTING===\n")
for i in range(0, len(testing_data)):
    result = network.feed_forward(testing_data[i]) \
        # if round(result, 1) == testing_outputs[i][0]:
    if get_rating(result) == get_rating(testing_outputs[i][0]):
         print("ROW %d - CORRECT" % i)
         print(result, testing_outputs[i][0])
         correct += 1
    else:
        print("ROW %d - WRONG!!! \n %f %f" % (i, result, testing_outputs[i][0]))
        print(result, testing_outputs[i][0])
        wrong += 1

print("CORRECT: %d WRONG: %d  ratio = %f" % (correct, wrong, correct / len(testing_data) * 100))


save_network_to_file(network, "network.txt")
network2 = load_network_from_file("network.txt")
print(network2.W1)
print(network2.W2)
print(network2.B1)
print(network2.B2)