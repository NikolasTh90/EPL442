import numpy as np
import NeuralNetwork
from time import time
def sigmoid(sum, a = 1):
    return 1.0/(1 + np.exp(-a*sum))

def feedForward(network:NeuralNetwork, dataInputs:list, patternNum:int):
    for i, layer in enumerate(network.getLayers()):
        if i == 0:
            j = 1 # changed to 1 because now the output is at the beginning
            for node in layer:
                if not node.isBias:
                    # print(f'input: {patternNum} {j} {list(dataInputs.values())[patternNum][j]}') # changed after reorganizing data
                    # print(dataInputs)
                    node.setOutput(dataInputs[patternNum][f'in{j}']) # changed after reorganizing data
                    j+=1
        else:
            # before = time()
            for node in layer:
                node.calculateOutput()
            # after = time()
            # print(f'Layer {i} calculated in {after-before} seconds')
    return [out.getOutput() for out in network.getLayers()[-1]]
            
        

