from Neuron import Neuron
import numpy as np
from matplotlib import pyplot as plt
import pickle
import sys
from multiprocessing import Pool, cpu_count
import time
import numpy as np
from Neuron import RBFNeuron, Neuron
import copy

class RBFNeuralNetwork:
    def __init__(self, numInputNeurons, numRBFNeurons, numOutputNeurons, centers, sigmas):
        self.numInputNeurons = numInputNeurons
        self.numRBFNeurons = numRBFNeurons
        self.numOutputNeurons = numOutputNeurons

        # Create RBF neurons with given centers and sigma
        if len(centers) <= numRBFNeurons: 
            self.rbfLayer = [RBFNeuron(center, sigma) for center, sigma in zip(centers, [sigmas for _ in range(len(centers))])]
            for i in range(len(centers), numRBFNeurons):
                self.rbfLayer.append(RBFNeuron(np.random.uniform(0, 1, numInputNeurons), sigmas))
        else:
            self.rbfLayer = [RBFNeuron(center, sigma) for center, sigma in zip(centers[:numRBFNeurons], [sigmas for _ in range(numRBFNeurons)])]

        # create the rbf neurons with random centers and given sigmas
        # self.rbfLayer = [RBFNeuron(np.random.uniform(0, 1, numInputNeurons), sigmas) for _ in range(numRBFNeurons)]

        self.rbfLayer.append(Neuron(isBias=True))

        # Create output neurons
        self.outputLayer = [Neuron(isOutput=True) for _ in range(numOutputNeurons)]

        # Connect RBF neurons to output neurons
        for rbfNeuron in self.rbfLayer:
            rbfNeuron.setConnectedToNeurons(self.outputLayer)

        # Initialize weights for connections between RBF neurons and output neurons
        for outputNeuron in self.outputLayer:
            outputNeuron.setConnectedFromNeurons(self.rbfLayer)
            outputNeuron.setWeights(np.random.uniform(0, 1, len(outputNeuron.connectedFromNeurons)))
            outputNeuron.weightsOld = copy.copy(outputNeuron.getWeights())

    def forwardPass(self, inputs):
        # Set inputs for RBF neurons
        for rbfNeuron in self.rbfLayer:
            if not rbfNeuron.isBias:
                rbfNeuron.setInputs(inputs)
        
        # Compute outputs of RBF layer
        rbfOutputs = np.array([neuron.getOutput() for neuron in self.rbfLayer])

        # Compute final outputs
        finalOutputs = []
        for outputNeuron in self.outputLayer:
            weightedSum = np.dot(rbfOutputs, outputNeuron.getWeights())
            outputNeuron.setOutput(weightedSum)  # Assuming linear output neuron
            finalOutputs.append(outputNeuron.getOutput())

        return finalOutputs

    def train(self, trainingInputs, trainingTargets, learningRates):
        centerLR = float(learningRates[0])
        weightLR = float(learningRates[1])
        sigmaLR =  float(learningRates[2])
        for inputVector, target in zip(trainingInputs, trainingTargets):
            # Forward pass
            outputs = self.forwardPass(inputVector)

            # Calculate error
            error = target - outputs
            # totalError += np.sum(error ** 2)
            # Update weights of output neurons
            self.updateOutputWeights(error, weightLR)
            # Update RBF neurons (centers and sigmas)
            self.updateRBFNeurons(inputVector, error, centerLR, sigmaLR)

            


    def updateRBFNeurons(self, inputVector, error, centerLR, sigmaLR):
        for rbfNeuron in self.rbfLayer:
            if not rbfNeuron.isBias:
                gaussian_output = rbfNeuron.getOutput()
                # for i in range(len(rbfNeuron.center)):
                sum_gradient = error * (inputVector - rbfNeuron.center) / (rbfNeuron.sigma ** 2)
                rbfNeuron.center += centerLR * sum_gradient * gaussian_output

                distance_squared = np.linalg.norm(inputVector - rbfNeuron.center) ** 2
                sum_gradient = error * distance_squared / (rbfNeuron.sigma ** 3)
                rbfNeuron.sigma += sigmaLR * sum_gradient * gaussian_output

    def updateOutputWeights(self, error, learningRate):
        for outputNeuron in self.outputLayer:
            for i, rbfNeuron in enumerate(self.rbfLayer):
                gaussian_output = rbfNeuron.getOutput() if not rbfNeuron.isBias else 1
                outputNeuron.weights[i] += learningRate * error * gaussian_output

    
    # Serializes and saves the entire object state
    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
