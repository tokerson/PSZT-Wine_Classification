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
        self.inputs = np.random.randint(0 , 11,  self.number_of_inputs) 
        self.hidden_layer = [ { 'weights': np.random.uniform(0.0, 1.0, self.number_of_inputs), 
                                'bias': np.random.uniform()} for i in range(self.hidden_neurons) ]
        
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    def calculate_the_value(self):
        return self.inputs[0] * self.hidden_layer[0]['weights'][0] + self.hidden_layer[0]['bias']


network = Network()
print(network.inputs[0])
print(network.hidden_layer[0]['weights'][0])
print(network.hidden_layer[0]['bias'])

print(network.calculate_the_value())
# print(network.inputs)
# print(network.hidden_layer[0])
# print(network.hidden_layer[0]['weights'][0])
    

