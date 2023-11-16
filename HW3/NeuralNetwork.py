from Neuron import Neuron
import numpy as np
from matplotlib import pyplot as plt
import pickle
import sys
from multiprocessing import Pool, cpu_count
import time
class SOMNeuralNetwork:
    # Constructor: Initializes the neural network with provided parameters
    def __init__(self, parameters: dict):
        self.parameters = parameters  # Store the given parameters
        # Initialize various attributes based on the parameters
        self.gridSize = int(parameters['gridSize'])
        self.numEpochs = int(parameters['numEpochs'])
        self.learningRate = parameters["learningRate"]
        self.startLearningRate = parameters["learningRate"]
        self.numInputNeurons = int(parameters['numInputNeurons'])
        self.gaussianRadius = parameters["gridSize"]/2.0  # Initial Gaussian radius
        self.currentPattern = 0  # Initial pattern index
        self.currentEpoch = 0    # Initial epoch index
        self.inputNeurons = self.createInputNeurons()  # Create input neurons
        self.TrainingError = []  # List to store training errors
        self.TestingError = []   # List to store testing errors
        self.letterMap = None    # Initialize letter mapping
        self.initializeWeights() # Initialize weights of neurons

    # Appends a training error to the list
    def appendTrainingError(self, trainingError):
        self.TrainingError.append(trainingError)
    
    # Appends a testing error to the list
    def appendTestingError(self, testingError):
        self.TestingError.append(testingError)

    # Creates and returns input neurons
    def createInputNeurons(self):
        inputNeurons = []
        for i in range(self.numInputNeurons):
            inputNeurons.append(Neuron())
        return inputNeurons

    # Initializes the weights for each input neuron
    def initializeWeights(self):
        for inputNeuron in self.inputNeurons:
            inputNeuron.setWeights(np.random.uniform(0, 1, (self.gridSize, self.gridSize)))

    # Sets inputs for each of the input neurons
    def setInputs(self, inputs):
        for i, inputNeuron in enumerate(self.inputNeurons):
            inputNeuron.setOutput(inputs[f'in{i+1}'])

    # Finds and returns the Best Matching Unit (BMU) for the current input
    def findBMU(self):
        input_outputs = np.array([neuron.getOutput() for neuron in self.inputNeurons])
        input_weights = np.array([neuron.getWeights() for neuron in self.inputNeurons])
        distances = np.sum((input_outputs[:, np.newaxis, np.newaxis] - input_weights) ** 2, axis=0)
        return np.unravel_index(np.argmin(distances), distances.shape)

    # Updates the learning rate based on the current epoch
    def updateLearningRate(self, epoch=None):
        if not epoch:
            epoch = self.currentEpoch
        self.learningRate = self.startLearningRate * np.exp(-epoch / self.numEpochs)
    
    # Updates the Gaussian radius based on the current epoch
    def updateGaussianRadius(self, epoch=None):
        if not epoch:
            epoch=self.currentEpoch
        self.gaussianRadius = self.parameters["gridSize"]/2.0 * np.exp(-epoch / self.numEpochs)
    
    # Updates the weights of the neurons based on the BMU
    def updateWeights(self, bmu):
        for inputNeuron in self.inputNeurons:
            i, j = np.meshgrid(range(self.gridSize), range(self.gridSize), indexing='ij')
            distance = np.sqrt((bmu[0] - i) ** 2 + (bmu[1] - j) ** 2)
            mask = distance <= self.gaussianRadius
            influence = np.exp(-distance ** 2 / (2 * self.gaussianRadius ** 2))
            delta_weights = self.learningRate * influence * (inputNeuron.getOutput() - inputNeuron.getWeights())
            inputNeuron.setWeights(inputNeuron.getWeights() + mask * delta_weights)

    # Maps each letter to a position in the grid
    def mapLetters(self, input_data):
        before = time.time()
        letterMap = np.zeros((self.gridSize, self.gridSize), dtype=object)

        for i in range(self.gridSize):
            for j in range(self.gridSize):
                print(f"Node ({i},{j})")
                minDistanceLetter = ''
                minDistance = sys.maxsize
                distance = 0

                for input in range(len(input_data)):
                    distance = np.sum((input_data[input][f'in{k}'] - self.inputNeurons[k - 1].getWeights()[i][j]) ** 2 for k in range(1, 17))
                    if distance < minDistance:
                        minDistance = distance
                        minDistanceLetter = input_data[input]['out']

                letterMap[i][j] = minDistanceLetter
        self.letterMap = letterMap
        after = time.time()
        print(f"Time taken: {after - before}")

    # Visualizes the SOM map
    def visualize_map(self):
        plt.figure(figsize=(self.gridSize, self.gridSize))
        for x in range(self.gridSize):
            for y in range(self.gridSize):
                try:
                    plt.text(x+0.2 if x<self.gridSize-1 else x-0.2, y+0.2 if y<self.gridSize-1 else y-0.2, self.letterMap[x, y] if self.letterMap[x, y] else ' ', va='center', ha='center')
                except:
                    plt.text(x+0.2 if x<self.gridSize-1 else x-0.2, y+0.2 if y<self.gridSize-1 else y-0.2, ' ', va='center', ha='center')

        plt.xticks(range(self.gridSize))
        plt.yticks(range(self.gridSize))
        plt.grid()
        plt.title("SOM Classification of Alphabet Letters")
        plt.show()

    # Visualizes the SOM map and prints it to a text file
    def visualize_to_txt(self, filename='clustering.txt'):
        with open(filename, 'w') as file:
            for x in range(self.gridSize):
                for y in range(self.gridSize):
                    try:
                        file.write(self.letterMap[x, y] if self.letterMap[x, y] else ' ')
                    except:
                        file.write(' ')
                    file.write(' ' if y < self.gridSize - 1 else '\n')

        print(f"Clustering results saved to {filename}")

    # Plots the training error over epochs
    def plotTrainingError(self):
        plt.plot(self.TrainingError)
        plt.title("Training Error")
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.show()
    
    # Plots the testing error over epochs
    def plotTestingError(self):
        plt.plot(self.TestingError)
        plt.title("Testing Error")
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.show()

    # Returns the grid of the SOM
    def getGrid(self):
        return self.grid
    
    # Returns the list of input neurons
    def getInputNeurons(self):
        return self.inputNeurons

    # Returns the current learning rate
    def getLearningRate(self):
        return self.learningRate
    
    # Serializes and saves the entire object state
    def save(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    # Provides a string representation of the SOMNeuralNetwork
    def __str__(self):
        nodes = ""
        for input in self.inputNeurons:
            nodes += "Neuron: " + str(input.id) + "\n"
            nodes += "Weights: " + str(input.getWeights()) + "\n"
            nodes += "Connected To: " + str(input.getConnectedToNeurons()) + "\n"
        nodes += "\n------------------------\n"
            
        return "Neural Network: \n" + \
            "Parameters: " + str(self.parameters) + "\n" + \
            "Nodes: " + "\n" + nodes
