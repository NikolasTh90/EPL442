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
    
    # def update_weights_batch(self, batch_indices, learning_rate, momentum):
    #     for i in batch_indices:
    #         weight = self.weights[i]
    #         delta = self.connectedToNeurons[i].delta
    #         output = self.output

    #         new_weight = weight - learning_rate * delta * output + momentum * (weight - self.weightsOld[i])
    #         self.weightsOld[i] = self.weights[i]
    #         self.weights[i] = new_weight

    # def updateWeights(self, learningRate, momentum):
    #     batch_size = 16  # Batch size for updating weights
    #     num_batches = len(self.weights) // batch_size


    #     with concurrent.futures.ProcessPoolExecutor() as executor:
    #         futures = []

    #         for i in range(num_batches):
    #             # Compute the indices for the current batch
    #             batch_indices = range(i * batch_size, (i + 1) * batch_size)

    #             # Update weights in parallel for the current batch
    #             future = executor.submit(self.update_weights_batch, batch_indices, learningRate, momentum)
    #             futures.append(future)
    #         # Handle any remaining weights after the last complete batch
    #         remaining_indices = range(num_batches * batch_size, len(self.weights))
    #         if remaining_indices:
    #             self.update_weights_batch(remaining_indices, learningRate, momentum)
    #         # Wait for all tasks to complete
    #         concurrent.futures.wait(futures)

    def updateWeights(self, learningRate, momentum):
        for i, weight in enumerate(self.weights):
            weight = weight - learningRate * self.connectedToNeurons[i].delta * self.output + momentum * (weight - self.weightsOld[i])
            self.weightsOld[i] = self.weights[i]
            self.weights[i] = weight


        


        
