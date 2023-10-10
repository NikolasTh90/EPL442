import numpy as np
from FeedForward import sigmoid
class Neuron():
    id = 0
    def __init__(self, isBias=False, isOutput=False):
        self.id = Neuron.id = Neuron.id + 1
        self.connectedFromNeurons = []
        self.connectedToNeurons = []
        self.output = 1
        self.error = 0
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
            "Connected From Neurons: " + str([neuron.id for neuron in self.connectedFromNeurons]) + "\n" + \
            "Connected To Neurons: " + str([neuron.id for neuron in self.connectedToNeurons]) + "\n" + \
            "Output: " + str(self.output) + "\n" + \
            "Error: " + str(self.error) + "\n" + \
            "Delta: " + str(self.delta) + "\n" + \
            "Weights: " + str(self.weights) + "\n"


    def calculateOutput(self):
        if self.isBias:
            self.output = 1
        else:
            inputs = np.array([])
            for prev_node in self.connectedFromNeurons:
                indexOfself = prev_node.connectedToNeurons.index(self)
                np.append(inputs, float(prev_node.getWeights()[indexOfself])*float(prev_node.getOutput()))
            
            self.output = sigmoid(inputs.sum())
        
        return self.output

    def calculateDelta(self, targetOutput = None):
        if self.isOutput:
            self.delta = self.output * (1 - self.output) * (self.output - targetOutput)
        else:
            self.delta = self.output * (1 - self.output) * sum([next_node.getDelta() * self.weights[i] for i, next_node in enumerate(self.connectedToNeurons)])

        return self.delta

    def updateWeights(self, learningRate, momentum):
        for i, weight in enumerate(self.weights):
            weight = weight - learningRate * self.delta * self.output + momentum * (self.weights[i] - self.weightsOld[i])
            self.weightsOld[i] = self.weights[i]
            self.weights[i] = weight


        


        
