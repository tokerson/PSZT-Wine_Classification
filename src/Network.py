import numpy as np

class Network:
    def __init__(self, inputs, hiddenLayers):
        self.inputs = inputs
        self.hiddenLayers = hiddenLayers
    
network = Network([
    [0,0],[0,1]
], 3)

print(network.inputs)
print(network.hiddenLayers)
    

