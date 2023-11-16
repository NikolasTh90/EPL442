import numpy as np
from time import time
import multiprocessing, concurrent.futures
class Neuron():
    id = 0  # Class variable to keep track of the last assigned neuron ID

    # Constructor to initialize a neuron with optional bias and output flags
    def __init__(self, isBias=False, isOutput=False):
        self.id = Neuron.id = Neuron.id + 1  # Assign a unique ID to the neuron and increment the class ID counter
        self.connectedFromNeurons = []  # Neurons from which this neuron receives input
        self.connectedToNeurons = []    # Neurons to which this neuron sends output
        self.output = 1  # Default output value
        self.delta = 0   # Delta value for backpropagation
        self.weights = []    # Weights of the incoming connections
        self.weightsOld = [] # Storage for old weights (not used in current implementation)
        self.isBias = isBias # Flag to determine if this neuron is a bias neuron
        self.isOutput = isOutput # Flag to determine if this neuron is an output neuron

    # Add a neuron from which this neuron receives input
    def addConnectedFromNeuron(self, neuron):
        self.connectedFromNeurons.append(neuron)
    
    # Add a neuron to which this neuron sends output
    def addConnectedToNeuron(self, neuron):
        self.connectedToNeurons.append(neuron)
    
    # Set the list of neurons to which this neuron sends output
    def setConnectedToNeurons(self, neurons:list):
        self.connectedToNeurons = neurons
    
    # Set the output value of the neuron
    def setOutput(self, output):
        self.output = output
        
    # Get the current output value of the neuron
    def getOutput(self):
        return self.output
    
    # Set the error value of the neuron (not used in current implementation)
    def setError(self, error):
        self.error = error
    
    # Get the current error value of the neuron
    def getError(self):
        return self.error
    
    # Set the delta value for backpropagation
    def setDelta(self, delta):
        self.delta = delta
    
    # Get the current delta value
    def getDelta(self):
        return self.delta
    
    # Set the weights of incoming connections
    def setWeights(self, weights):
        self.weights = weights

    # Set a specific weight in the weights matrix
    def setWeight(self, i, j, weight):
        self.weights[i][j] = weight
    
    # Get the current weights of incoming connections
    def getWeights(self):
        return self.weights
    
    # Get the list of neurons from which this neuron receives input
    def getConnectedFromNeurons(self):
        return self.connectedFromNeurons
    
    # Get the list of neurons to which this neuron sends output
    def getConnectedToNeurons(self):
        return self.connectedToNeurons
    
    # String representation of the neuron, showing various attributes
    def __str__(self):
        return "Neuron: " + str(self.id) + "\n" + \
            "is Output: " + str(self.isOutput) + "\n" + \
            "is Bias: " + str(self.isBias) + "\n" + \
            "Connected From Neurons: " + str([neuron.id for neuron in self.connectedFromNeurons]) + "\n" + \
            "Connected To Neurons: " + str([neuron.id for neuron in self.connectedToNeurons]) + "\n" + \
            "Output: " + str(self.output) + "\n" + \
            "Delta: " + str(self.delta) + "\n" + \
            "Weights: " + str(self.weights) + "\n"



