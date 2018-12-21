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
        # print("B1")
        # print(self.B1)
        self.Z1 = np.dot(input, self.W1) + self.B1
        # print("Z1")
        # print(self.Z1)
        
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.B2
        output = self.sigmoid(self.Z2)

        return output

    def backward(self,X, expected, calculated ):
        self.loss = (expected - calculated)      # (n x 1)

        self.dZ2 = self.loss * self.sigmoidPrime(calculated)
        self.dW2 = np.dot(self.A1.T, self.dZ2) / len(X)                    # (n x h)^T(n x 1) = (h x n)(n x 1)= (h x 1)
        self.dB2 = np.sum(self.loss.T, axis = 1, keepdims = True) / len(X)    
        
        self.dZ1 = np.dot(self.dZ2, self.W2.T)*self.sigmoidPrime(self.A1)   # (n x h) = (n x 1)(h x 1)^ T = (n x 1)(1 x h) 
        self.dW1 = np.dot(X.T, self.dZ1) / len(X)                              # ( m x h) = ( n x m)^T(n x h) = (m x n)(n x h)
        # print("dZ1")
        # print(self.dZ1)
        self.dB1 = np.sum(self.dZ1.T, axis = 1, keepdims = True) / len(X)     # ( 1 x h)
        
        self.W1 += self.learningRate*self.dW1
        self.W2 += self.learningRate*self.dW2
        # print("B1")
        # print(self.B1)
        # print("dB1")
        # print(self.dB1)
        self.B1 += self.learningRate*self.dB1.T
        self.B2 += self.learningRate*self.dB2


    def train (self, X , Y):
        output = self.feed_forward(X)
        print("Output")
        print(output)
        print("Expected output")
        print(Y)
        self.backward(X,Y,output)
    

network = Network()
X = np.array((
    [ 0, 1 , 2, 3, 4, 5, 6, 7, 8, 9 , 10, 11 ],
    [ 1, 3, 4 ,5 ,6 ,7, 1, 2, 4, 5, 2 ,1],
    [ 0, 1 , 2, 3, 4, 5, 6, 7, 8, 9 , 10, 11 ],
    [ 1 ,2 , 3 ,1 ,2 ,4 ,5 ,0.5 ,6 ,2 , 4 ,1],
    [ 1 ,5 , 1 ,2 ,2 ,4 ,5 ,1.5 ,6 ,2 , 4 ,1]
), dtype=float)
Y = np.array((
      [0.5], 
      [0.35],
      [0.5],
      [0.8],
      [0.63]
), dtype=float)  
for i in range(5000):
    network.train(X,Y)

print("B1")
print(network.B1)
print("B2")
print(network.B2)

print("check")
print(network.feed_forward([[ 0, 1 , 2, 3, 4, 5, 6, 7, 8, 9 , 10, 11 ]]))

