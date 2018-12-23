import numpy as np

# from src.CsvReader import get_data, normalize_data, seperate_inputs_and_outputs
from src.CsvReader import *


class Network:

    def __init__(self, inputSize=11, hiddenSize=4):
        self.outputSize = 1
        self.learningRate = 0.05
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
        h = np.matrix([
            [output_loss*self.W2[0, 0] * self.sigmoid_derivative(self.A1[0, 0]),
             output_loss * self.W2[1, 0] * self.sigmoid_derivative(self.A1[0, 1]),
             output_loss * self.W2[2, 0] * self.sigmoid_derivative(self.A1[0, 2]),
             output_loss * self.W2[3, 0] * self.sigmoid_derivative(self.A1[0, 3])]
        ])
        hidden_loss = h.getA()

        update_W1 = np.matrix([
            [hidden_loss[0][0] * input[0], hidden_loss[0][1] * input[0], hidden_loss[0][2] * input[0],
             hidden_loss[0][3] * input[0]],
            [hidden_loss[0][0] * input[1], hidden_loss[0][1] * input[1], hidden_loss[0][2] * input[1],
             hidden_loss[0][3] * input[1]],
            [hidden_loss[0][0] * input[2], hidden_loss[0][1] * input[2], hidden_loss[0][2] * input[2],
             hidden_loss[0][3] * input[2]],
            [hidden_loss[0][0] * input[3], hidden_loss[0][1] * input[3], hidden_loss[0][2] * input[3],
             hidden_loss[0][3] * input[3]],
            [hidden_loss[0][0] * input[4], hidden_loss[0][1] * input[4], hidden_loss[0][2] * input[4],
             hidden_loss[0][3] * input[4]],
            [hidden_loss[0][0] * input[5], hidden_loss[0][1] * input[5], hidden_loss[0][2] * input[5],
             hidden_loss[0][3] * input[5]],
            [hidden_loss[0][0] * input[6], hidden_loss[0][1] * input[6], hidden_loss[0][2] * input[6],
             hidden_loss[0][3] * input[6]],
            [hidden_loss[0][0] * input[7], hidden_loss[0][1] * input[7], hidden_loss[0][2] * input[7],
             hidden_loss[0][3] * input[7]],
            [hidden_loss[0][0] * input[8], hidden_loss[0][1] * input[8], hidden_loss[0][2] * input[8],
             hidden_loss[0][3] * input[8]],
            [hidden_loss[0][0] * input[9], hidden_loss[0][1] * input[9], hidden_loss[0][2] * input[9],
             hidden_loss[0][3] * input[9]],
            [hidden_loss[0][0] * input[10], hidden_loss[0][1] * input[10], hidden_loss[0][2] * input[10],
             hidden_loss[0][3] * input[10]]
        ])

        self.W1 = (np.asmatrix(self.W1) + update_W1 * self.learningRate).getA()

        update_B1 = np.matrix([
            [hidden_loss[0][0], hidden_loss[0][1], hidden_loss[0][2], hidden_loss[0][3]]
        ])

        self.B1 = (np.asmatrix(self.B1) + update_B1 * self.learningRate).getA()

        update_W2 = np.matrix([
            [output_loss * self.A1[0][0]],
            [output_loss * self.A1[0][1]],
            [output_loss * self.A1[0][2]],
            [output_loss * self.A1[0][3]]
        ])

        self.W2 = (np.asmatrix(self.W2) + update_W2 * self.learningRate).getA()

        update_B2 = np.matrix([
            [output_loss]
        ])

        self.B2 = (np.asmatrix(self.B2) + update_B2 * self.learningRate).getA()

    # def backward(self, X, expected, calculated):
    #
    #     # self.loss = (expected - calculated)  # (n x 1)
    #     #
    #     # self.dZ2 = self.loss * self.sigmoidPrime(calculated)
    #     # self.dW2 = np.dot(self.A1.T, self.dZ2) / len(X)  # (n x h)^T(n x 1) = (h x n)(n x 1)= (h x 1)
    #     # self.dB2 = np.sum(self.loss.T, axis=1, keepdims=True) / len(X)
    #     #
    #     # self.dZ1 = np.dot(self.dZ2, self.W2.T) * self.sigmoidPrime(self.A1)  # (n x h) = (n x 1)(h x 1)^ T = (n x 1)(1 x h)
    #     # self.dW1 = np.dot(X.T, self.dZ1) / len(X)  # ( m x h) = ( n x m)^T(n x h) = (m x n)(n x h)
    #     # self.dB1 = np.sum(self.dZ1.T, axis=1, keepdims=True) / len(X)  # ( 1 x h)
    #     #
    #     # self.W1 += self.learningRate * self.dW1
    #     # self.W2 += self.learningRate * self.dW2
    #     #
    #     # self.B1 += self.learningRate * self.dB1.T
    #     # self.B2 += self.learningRate * self.dB2
    #
    # def train(self, X, Y):
    #     output = self.feed_forward(X)
    #     print("Output")
    #     print(output)
    #     print("Expected output")
    #     print(Y)
    #     self.backward(X, Y, output)


# network = Network()
# X = np.array((
#     [ 0, 1 , 2, 3, 4, 5, 6, 7, 8, 9 , 10, 11 ],
#     [ 1, 3, 4 ,5 ,6 ,7, 1, 2, 4, 5, 2 ,1],
#     [ 0, 1 , 2, 3, 4, 5, 6, 7, 8, 9 , 10, 11 ],
#     [ 1 ,2 , 3 ,1 ,2 ,4 ,5 ,0.5 ,6 ,2 , 4 ,1],
#     [ 1 ,5 , 1 ,2 ,2 ,4 ,5 ,1.5 ,6 ,2 , 4 ,1]
# ), dtype=float)
# Y = np.array((
#       [0.5],
#       [0.35],
#       [0.5],
#       [0.8],
#       [0.63]
# ), dtype=float)
# for i in range(5000):
#     network.train(X,Y)
#
# print("B1")
# print(network.B1)
# print("B2")
# print(network.B2)
#
# print("check")
# print(network.feed_forward([[ 0, 1 , 2, 3, 4, 5, 6, 7, 8, 9 , 10, 11 ]]))

# network = Network()
# X = np.array((
#     [ 0, 1 , 2, 3, 4, 5, 6, 7, 8, 9 , 10, 11 ],
#     [ 1, 3, 4 ,5 ,6 ,7, 1, 2, 4, 5, 2 ,1],
#     [ 0, 1 , 2, 3, 4, 5, 6, 7, 8, 9 , 10, 11 ],
#     [ 1 ,2 , 3 ,1 ,2 ,4 ,5 ,0.5 ,6 ,2 , 4 ,1],
#     [ 1 ,5 , 1 ,2 ,2 ,4 ,5 ,1.5 ,6 ,2 , 4 ,1]
# ), dtype=float)
# Y = np.array((
#       [0.5],
#       [0.35],
#       [0.5],
#       [0.8],
#       [0.63]
# ), dtype=float)
# for i in range(5000):
#     network.train(X,Y)


# network = Network()
# data = get_data(100, '../data/winequality-red.csv')
# outputs = []
# normalize_data(data)
# # print(data)
# seperate_inputs_and_outputs(data, outputs)
# X = np.array((data), dtype=float)
# Y = np.array((outputs), dtype=float)
#
# # print(X)
# # print(Y)
#
# for i in range(100):
#     network.train(X, Y)


network = Network()
network.learningRate = 0.4
result = network.feed_forward([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]) #expected 5
print(result)
network.backward_propagation(0.5, result[0][0], [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4])
result = network.feed_forward([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]) #expected 5
print(result)
network.backward_propagation(0.5, result[0][0], [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4])
result = network.feed_forward([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]) #expected 5
print(result)
network.backward_propagation(0.5, result[0][0], [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4])
result = network.feed_forward([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]) #expected 5
print(result)
network.backward_propagation(0.5, result[0][0], [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4])
result = network.feed_forward([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]) #expected 5
print(result)
network.backward_propagation(0.5, result[0][0], [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4])
result = network.feed_forward([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]) #expected 5
print(result)
network.backward_propagation(0.5, result[0][0], [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4])
result = network.feed_forward([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]) #expected 5
print(result)
network.backward_propagation(0.5, result[0][0], [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4])
result = network.feed_forward([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]) #expected 5
print(result)
network.backward_propagation(0.5, result[0][0], [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4])
result = network.feed_forward([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]) #expected 5
print(result)
network.backward_propagation(0.5, result[0][0], [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4])
result = network.feed_forward([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]) #expected 5
print(result)
network.backward_propagation(0.5, result[0][0], [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4])
result = network.feed_forward([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]) #expected 5
print(result)
network.backward_propagation(0.5, result[0][0], [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4])
result = network.feed_forward([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]) #expected 5
print(result)
network.backward_propagation(0.5, result[0][0], [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4])
result = network.feed_forward([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]) #expected 5
print(result)
network.backward_propagation(0.5, result[0][0], [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4])
result = network.feed_forward([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]) #expected 5
print(result)
network.backward_propagation(0.5, result[0][0], [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4])
result = network.feed_forward([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]) #expected 5
print(result)
network.backward_propagation(0.5, result[0][0], [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4])
result = network.feed_forward([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]) #expected 5
print(result)
network.backward_propagation(0.5, result[0][0], [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4])
result = network.feed_forward([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]) #expected 5
print(result)
network.backward_propagation(0.5, result[0][0], [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4])
result = network.feed_forward([7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]) #expected 5
print(result)
