import numpy as np
from time import time
import multiprocessing, concurrent.futures
class Neuron():
    id = 0
    def __init__(self, isBias=False, isOutput=False):
        self.id = Neuron.id = Neuron.id + 1
        self.connectedFromNeurons = []
        self.connectedToNeurons = []
        self.output = 1
        self.delta = 0
        self.weights = []
        self.weightsOld = []
        self.isBias = isBias
        self.isOutput = isOutput



    def addConnectedFromNeuron(self, neuron):
        self.connectedFromNeurons.append(neuron)
    
    def addConnectedToNeuron(self, neuron):
        self.connectedToNeurons.append(neuron)
    
    def setConnectedToNeurons(self, neurons:list):
        self.connectedToNeurons = neurons
    
    def setOutput(self, output):
        self.output = output
        
    def getOutput(self):
        return self.output
    
    def setError(self, error):
        self.error = error
    
    def getError(self):
        return self.error
    
    def setDelta(self, delta):
        self.delta = delta
    
    def getDelta(self):
        return self.delta
    
    def setWeights(self, weights):
        self.weights = weights

    def setWeight(self, i, j, weight):
        self.weights[i][j] = weight
    
    def getWeights(self):
        return self.weights
    
    def getConnectedFromNeurons(self):
        return self.connectedFromNeurons
    
    def getConnectedToNeurons(self):
        return self.connectedToNeurons
    
    def __str__(self):
        return "Neuron: " + str(self.id) + "\n" + \
            "is Output: " + str(self.isOutput) + "\n" + \
            "is Bias: " + str(self.isBias) + "\n" + \
            "Connected From Neurons: " + str([neuron.id for neuron in self.connectedFromNeurons]) + "\n" + \
            "Connected To Neurons: " + str([neuron.id for neuron in self.connectedToNeurons]) + "\n" + \
            "Output: " + str(self.output) + "\n" + \
            "Delta: " + str(self.delta) + "\n" + \
            "Weights: " + str(self.weights) + "\n"


