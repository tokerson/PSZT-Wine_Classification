import numpy as np

#class describing how our network looks like
#Fields:    inputs - array of 12 Integers from 0 to 10
#           hidden_neurons - size of hidden_layer ( number of hidden neurons)
#           hidden_layer - array containing objects with two fields ( 'weights' is an array of weights of the edges coming from the input,
#                                                                     'bias' is a value added in neuron)
#           hidden_layer[2]['weights'][0] is a weight of the edge starting in the 1st input neuron and ending in the 3rd hidden neuron
#            
class Network:

    #user specyfies number of input nodes and how big should the hidden layer be
    def __init__(self, hidden_neurons = 20):
        self.number_of_inputs = 12
        self.hidden_neurons = hidden_neurons
        self.inputs = np.random.randint(low = 0 , high = 11, size = self.number_of_inputs) # array of random integers from 0 to 10 
        self.hidden_layer = [ { 'weights': np.random.uniform(low = -1.0 / np.sqrt(self.number_of_inputs), 
                                                             high = 1.0 / np.sqrt(self.number_of_inputs), 
                                                             size = self.number_of_inputs),  
                                'bias': np.random.uniform()} for i in range(self.hidden_neurons) ]
        self.output_neuron =  { 'weights':  np.random.uniform(low = -1.0 / np.sqrt(self.hidden_neurons), 
                                                             high = 1.0 / np.sqrt(self.hidden_neurons)  , 
                                                             size = self.hidden_neurons),
                                 'bias' : np.random.uniform()}
        
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def feed_forward(self):
        output = 0.0
        result = [0] * self.hidden_neurons  #creating an array of size equals to hidden neurons number and filled with zeros.
        #foreach hidden neuron we are calculating its output based on the weights of the edges
        for j in range(0, self.hidden_neurons):

            for i in range(0, self.number_of_inputs):
                result[j] += self.inputs[i] * self.hidden_layer[j]['weights'][i] 
            
            result[j] += self.hidden_layer[j]['bias']
            result[j] = self.sigmoid(result[j])
        
        for i in range(0, self.hidden_neurons):
            output += result[i] * self.output_neuron['weights'][i]

        output += self.output_neuron['bias']
        return self.sigmoid(output)

network = Network()

print("inputs:")
print(network.inputs)
print("weights:")
for neuron in network.hidden_layer:
    print("Weights: \n", neuron['weights'])
    print("Bias: ", neuron['bias'])
    
print("Output", network.feed_forward())
# print("Bias:", network.hidden_layer[0]['bias'])

# print("Result:", network.feed_forward(0))
# print(network.inputs)
# print(network.hidden_layer)
# print(network.hidden_layer[0]['weights'][0])
    

