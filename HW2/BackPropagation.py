import concurrent.futures
import copyreg
import types
import multiprocessing
from functools import partial
import NeuralNetwork
from utility import TARGET_OUTPUTS
from time import time
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
    outputs = [1 if TARGET_OUTPUTS.index(data[patternNum]['out']) == i else 0 for i in range(len(TARGET_OUTPUTS))] 
    calculateDeltas(network, outputs)
    updateWeights(network)