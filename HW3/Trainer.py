from math import sqrt
from FileManager import FileReader, FileWriter
from NeuralNetwork import SOMNeuralNetwork
import numpy as np
from utility import TARGET_OUTPUTS, prepare_data
import time
import multiprocessing
import pickle
max_workers=multiprocessing.cpu_count()
print(max_workers)
def getParameters(filename='parameters.txt'):
    parameters = FileReader('parameters.txt').getParameters()
    return parameters

def load_som_model(filename):
    with open(filename, 'rb') as file:
        som = pickle.load(file)
    return som

def calculate_error(data, som):
    error = 0
    for i in range(len(data)):
        som.setInputs(data[i])
        bmu = som.findBMU()
        error += sum((data[i][f'in{k}'] - som.inputNeurons[k - 1].getWeights()[bmu[0]][bmu[1]]) ** 2 for k in range(1, 17))
    error /= len(data)
    
    return error

def train(filename='all_data.txt', epochs=None):
    parameters = getParameters()
    data = prepare_data(filename)
    print('Training Data Length:', len(data['trainingData']))
    try:
        SOM = load_som_model(f"som_LR={parameters['learningRate']}_GS={parameters['gridSize']}_EP={parameters['numEpochs']}.pkl")
    except:
        SOM = SOMNeuralNetwork(parameters)

    SOM.currentEpoch = SOM.currentEpoch
    try:
       while SOM.currentEpoch < int(parameters["numEpochs"]):
            print(f'Epoch {SOM.currentEpoch + 1}')
            start_time = time.time()
            while SOM.currentPattern < len(data['trainingData']):
                print(f"Progress: {SOM.currentPattern}/{len(data['trainingData'])}", end="\r")
                SOM.setInputs(data["trainingData"][SOM.currentPattern])
                bmu = SOM.findBMU()
                SOM.updateLearningRate()
                SOM.updateGaussianRadius()
                SOM.updateWeights(bmu)
                SOM.currentPattern += 1
            end_time = time.time()
            print(f'Epoch {SOM.currentEpoch + 1} completed in {end_time - start_time} seconds')
            
            start_time = time.time()
            train_error = calculate_error(data['trainingData'], SOM)
            SOM.appendTrainingError(train_error)
            end_time = time.time()
            print(f'Epoch {SOM.currentEpoch+1} training error calculated in {end_time - start_time} seconds')
            print(f"Train Error={train_error}")
            start_time = time.time()
            test_error = calculate_error(data['testData'], SOM)
            SOM.appendTestingError(test_error)
            end_time = time.time()
            print(f'Epoch {SOM.currentEpoch+1} testing error calculated in {end_time - start_time} seconds')
            print(f"Test Error={test_error}")
            SOM.currentPattern = 0
            SOM.currentEpoch += 1
            
    except KeyboardInterrupt:
        print("Training interrupted")
    
    print("Training complete")
    SOM.save(f"som_LR={SOM.parameters['learningRate']}_GS={SOM.parameters['gridSize']}_EP={parameters['numEpochs']}.pkl")
    SOM.plotTrainingError()
    SOM.plotTestingError()
    if not SOM.letterMap:
        SOM.mapLetters(data['testData'])
        SOM.save(f"som_LR={SOM.parameters['learningRate']}_GS={SOM.parameters['gridSize']}_EP={parameters['numEpochs']}.pkl")

    SOM.visualize_map()
    
np.random.seed(2)    
train()

