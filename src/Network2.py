import numpy as np

class Network:

    def __init__(self, inputSize = 12, hiddenSize = 4):
        self.outputSize = 1
        self.learningRate = 0.05
        self.W1 = np.random.randn(inputSize, hiddenSize) / np.sqrt(inputSize)
        self.W2 = np.random.randn(hiddenSize, self.outputSize) / np.sqrt(hiddenSize)
        self.B1 = np.zeros((1,hiddenSize))
        self.B2 = np.zeros((1,self.outputSize))

    def sigmoid(self, x):
        return 1 / ( 1 + np.exp(-x))
    
    def sigmoidPrime(self, x):
        return x * ( 1 - x)

    def feed_forward(self, input):
        self.Z1 = np.dot(input, self.W1) + self.B1
        self.Z2 = self.sigmoid(self.Z1)
        self.Z3 = np.dot(self.Z2, self.W2) + self.B2 
        output = self.sigmoid(self.Z3)
        print(output)
        return output

    def backward(self,X, expected, calculated ):
        self.loss = expected - calculated
        self.o_delta = self.loss*self.sigmoidPrime(calculated)

        self.Z2_error = self.o_delta.dot(self.W2.T)
        self.Z2_delta = self.Z2_error*self.sigmoidPrime(self.Z2)

        self.W1 += self.learningRate*X.T.dot(self.Z2_delta)
        self.W2 += self.learningRate*self.Z2.T.dot(self.o_delta)

        
    def train (self, X , Y):
        output = self.feed_forward(X)
        print("Output")
        print(output)
        print("Expected output")
        print(Y)
        self.backward(X,Y,output)
        print("W1")
        print(self.W1)
        print("W2")
        print(self.W2)

    

network = Network()
X = np.array((
    [ 0, 1 , 2, 3, 4, 5, 6, 7, 8, 9 , 10, 11 ],
    [ 1, 3, 4 ,5 ,6 ,7, 1, 2, 4, 5, 2 ,1],
    [ 0, 1 , 2, 3, 4, 5, 6, 7, 8, 9 , 10, 11 ],
    [ 1 ,2 , 3 ,1 ,2 ,4 ,5 ,0.5 ,6 ,2 , 4 ,1]
), dtype=float)
Y = np.array((
      [0.5], 
      [0.35],
      [0.5],
      [0.8]
), dtype=float)  
for i in range(10000):
    network.train(X,Y)
        
