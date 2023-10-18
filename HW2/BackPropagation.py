import NeuralNetwork
from Trainer import TARGET_OUPUTS
def calculateDeltas(network:NeuralNetwork,outputs):
    for layer in reversed(network.getLayers()):
        if layer == network.getLayers()[-1]:
            for i, node in enumerate(layer):
                node.calculateDelta(targetOutput = outputs[i])
        else:
            if layer == network.getLayers()[0]:
                continue
            for node in layer:
                node.calculateDelta()

def updateWeights(network:NeuralNetwork):
    for layer in network.getLayers():
        for node in layer:
            node.updateWeights(learningRate = network.getLearningRate(), momentum = network.getMomentum())

def backPropagation(network:NeuralNetwork, data:dict, patternNum:int):
    outputs = [1 if TARGET_OUPUTS.index(data['out'][patternNum]) == i else 0 for i in range(len(TARGET_OUPUTS))] 
    calculateDeltas(network, outputs)
    updateWeights(network)