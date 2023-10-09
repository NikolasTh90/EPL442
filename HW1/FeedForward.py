import numpy as np
import NeuralNetwork
def sigmoid(sum, a = 1):
    return 1.0/(1 + np.exp(-a*sum))

def feedForward(network:NeuralNetwork, dataInputs:dict, patternNum:int):
    for i, layer in enumerate(network.layers):
        if i == 0:
            for j, node in enumerate(layer):
                node.setOutput(dataInputs[j][patternNum])
        else:
            for node in layer:
                node.calulateOutput()
            
        

