import numpy as np
import NeuralNetwork
def sigmoid(sum, a = 1):
    return 1.0/(1 + np.exp(-a*sum))

def feedForward(network:NeuralNetwork, dataInputs:dict, patternNum:int):
    for i, layer in enumerate(network.getLayers()):
        if i == 0:
            j = 1 # changed to 1 because now the output is at the beginning
            for node in layer:
                if not node.isBias:
                    print(f'input: {patternNum} {j} {list(dataInputs.values())[j][patternNum]}')

                    node.setOutput(list(dataInputs.values())[j][patternNum])
                    j+=1
        else:
            for node in layer:
                node.calculateOutput()
    return [out.getOutput() for out in network.getLayers()[-1]]
            
        

