import numpy as np
from FeedForward import sigmoid
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


    def calculateOutput(self):
        if self.isBias:
            self.output = 1
        else:
            weighted_sum = np.dot([prev_node.getWeights()[prev_node.connectedToNeurons.index(self)] for prev_node in self.connectedFromNeurons],
                              [prev_node.getOutput() for prev_node in self.connectedFromNeurons])
            self.output = sigmoid(weighted_sum)
            
        
        return self.output

    def calculateDelta(self, targetOutput = None):
        if not self.isBias:
            if self.isOutput:
                self.delta = self.output * (1 - self.output) * (self.output - targetOutput)
            else:
                self.delta = self.output * (1 - self.output) * sum([next_node.getDelta() * self.weights[i] for i, next_node in enumerate(self.connectedToNeurons)])

        return self.delta
    

    def updateWeights(self, learningRate, momentum):
        weights = np.array(self.weights)
        weights_old = np.array(self.weightsOld)
        deltas = np.array([neuron.delta for neuron in self.connectedToNeurons])
        outputs = np.array(self.output)

        # Calculate weight updates
        weight_updates = learningRate * deltas * outputs + momentum * (weights - weights_old)

        # Update weights and weightsOld
        self.weights = weights - weight_updates
        self.weightsOld = weights

            


        
