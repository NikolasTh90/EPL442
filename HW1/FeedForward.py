import numpy as np
import NeuralNetwork
def sigmoid(sum, a = 1):
    return 1.0/(1 + np.exp(-a*sum))

def feedForward(network:NeuralNetwork, dataInputs:dict, patternNum:int):
    for i, layer in enumerate(network.getLayers()):
        if i == 0:
            for j, node in enumerate(layer):
                node.setOutput(list(dataInputs.values())[j][patternNum])
        else:
            for node in layer:
                node.calculateOutput()
    return network.getLayers()[-1][0].getOutput()
            
        

