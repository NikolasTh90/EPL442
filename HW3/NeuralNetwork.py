from Neuron import Neuron
import numpy as np
from matplotlib import pyplot as plt
import pickle
import sys
from multiprocessing import Pool, cpu_count
import time
class SOMNeuralNetwork:
    def __init__(self, parameters: dict):
        self.parameters = parameters
        self.gridSize = int(parameters['gridSize'])
        self.numEpochs = int(parameters['numEpochs'])
        self.learningRate = parameters["learningRate"]
        self.startLearningRate = parameters["learningRate"]
        self.numInputNeurons = int(parameters['numInputNeurons'])
        self.gaussianRadius = parameters["gridSize"]/2.0
        self.currentPattern = 0
        self.currentEpoch = 0
        self.inputNeurons = self.createInputNeurons()
        self.TrainingError = []
        self.TestingError = []
        self.letterMap = None
        self.initializeWeights()

    def appendTrainingError(self, trainingError):
        self.TrainingError.append(trainingError)
    
    def appendTestingError(self, testingError):
        self.TestingError.append(testingError)


    def createInputNeurons(self):
        inputNeurons = []
        for i in range(self.numInputNeurons):
            inputNeurons.append(Neuron())
        return inputNeurons

    def initializeWeights(self):
        for inputNeuron in self.inputNeurons:
            # Initialize weights for each input neuron
            inputNeuron.setWeights(np.random.uniform(0, 1, (self.gridSize, self.gridSize)))

    def setInputs(self, inputs):
        for i, inputNeuron in enumerate(self.inputNeurons):
            inputNeuron.setOutput(inputs[f'in{i+1}'])

    def findBMU(self):
        input_outputs = np.array([neuron.getOutput() for neuron in self.inputNeurons])
        input_weights = np.array([neuron.getWeights() for neuron in self.inputNeurons])
        distances = np.sum((input_outputs[:, np.newaxis, np.newaxis] - input_weights) ** 2, axis=0)
        return np.unravel_index(np.argmin(distances), distances.shape)

    
    def updateLearningRate(self, epoch=None):
        if not epoch:
            epoch = self.currentEpoch
        self.learningRate = self.startLearningRate * np.exp(-epoch / self.numEpochs)
    
    def updateGaussianRadius(self, epoch=None):
        if not epoch:
            epoch=self.currentEpoch
        self.gaussianRadius = self.parameters["gridSize"]/2.0 * np.exp(-epoch / self.numEpochs)
    

    def updateWeights(self, bmu):
        for inputNeuron in self.inputNeurons:
            i, j = np.meshgrid(range(self.gridSize), range(self.gridSize), indexing='ij')
            distance = np.sqrt((bmu[0] - i) ** 2 + (bmu[1] - j) ** 2)
            mask = distance <= self.gaussianRadius
            influence = np.exp(-distance ** 2 / (2 * self.gaussianRadius ** 2))
            delta_weights = self.learningRate * influence * (inputNeuron.getOutput() - inputNeuron.getWeights())
            inputNeuron.setWeights(inputNeuron.getWeights() + mask * delta_weights)


    def mapLetters(self, input_data):
        before = time.time()

        letterMap = np.zeros((self.gridSize, self.gridSize), dtype=object)

        # Find the winning neuron for each input and store it in the grid

        for i in range(self.gridSize):
            for j in range(self.gridSize):
                print(f"Node ({i},{j})")

                minDistanceLetter = ''
                minDistance = sys.maxsize
                distance = 0

                for input in range(len(input_data)):
                        # print(f"Input ({input})", end="\r")
                        distance = np.sum((input_data[input][f'in{k}'] - self.inputNeurons[k - 1].getWeights()[i][j]) ** 2 for k in range(1, 17))
                        if distance < minDistance:
                            minDistance = distance
                            minDistanceLetter = input_data[input]['out']

                letterMap[i][j] = minDistanceLetter
        self.letterMap = letterMap
        after = time.time()
        print(f"Time taken: {after - before}")

    # visualize the som map
    def visualize_map(self):
        
        
        # Create a plot to visualize the SOM
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

    # visualize the som map and print to text file
    def visualize_to_txt(self, filename='clustering.txt'):
        # Open the file in write mode
        with open(filename, 'w') as file:
            for x in range(self.gridSize):
                for y in range(self.gridSize):
                    try:
                        # Write the cluster labels to the file
                        file.write(self.letterMap[x, y] if self.letterMap[x, y] else ' ')
                    except:
                        # Handle the case when an exception occurs
                        file.write(' ')
                    # Add a space or newline to separate values
                    file.write(' ' if y < self.gridSize - 1 else '\n')

        print(f"Clustering results saved to {filename}")


    def plotTrainingError(self):
        plt.plot(self.TrainingError)
        plt.title("Training Error")
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.show()
    
    def plotTestingError(self):
        plt.plot(self.TestingError)
        plt.title("Testing Error")
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.show()


    def getGrid(self):
        return self.grid
    
    def getInputNeurons(self):
        return self.inputNeurons

    def getLearningRate(self):
        return self.learningRate
    
    def save(self, filename):
        # Serialize the entire object using pickle
        with open(filename, 'wb') as file:
            pickle.dump(self, file)


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
            
