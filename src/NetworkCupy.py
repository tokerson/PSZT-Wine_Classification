import cupy as cp


class NetworkCupy:

    def __init__(self, inputSize=11, hiddenSize=4):
        self.output_size = 1
        self.learningRate = 0.05
        self.input_size = inputSize
        self.hidden_size = hiddenSize
        self.W1 = cp.random.randn(inputSize, hiddenSize) / cp.sqrt(inputSize)
        self.W2 = cp.random.randn(hiddenSize, self.output_size) / cp.sqrt(hiddenSize)
        self.B1 = cp.zeros((1, hiddenSize))
        self.B2 = cp.zeros((1, self.output_size))

        self.X1 = cp.zeros((1, hiddenSize))
        self.X2 = cp.zeros((1, self.output_size))

        self.A1 = None

    def sigmoid(self, x):
        return 1 / (1 + cp.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feed_forward(self, input):
        cp.dot(cp.array(input), self.W1, self.X1)
        Z1 = cp.add(self.X1, self.B1)
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

        update_W1 = cp.zeros((self.input_size, self.hidden_size), cp.float32)
        for i in range(0, self.input_size):
            for j in range(0, self.hidden_size):
                update_W1[i][j] = hidden_loss[j] * input[i] * self.learningRate

        self.W1 = cp.add(self.W1, update_W1)

        update_B1 = cp.zeros(self.hidden_size, cp.float32)

        for i in range(0, self.hidden_size):
            update_B1[i] = hidden_loss[i] * self.learningRate

        self.B1 = cp.add(self.B1, update_B1)

        update_W2 = cp.zeros((self.hidden_size, self.output_size), cp.float32)

        for i in range(0, self.hidden_size):
            update_W2[i][0] = output_loss * self.A1[0][i]

        for i in range(0, self.hidden_size):
            for j in range(0, self.output_size):
                update_W2 *= self.learningRate

        self.W2 = cp.add(self.W2, update_W2)

        update_B2 = cp.array(output_loss)

        self.B2 = cp.add(self.B2, (update_B2 * self.learningRate))
