from math import sqrt
from FileManager import FileReader, FileWriter
from NeuralNetwork import SOMNeuralNetwork
import numpy as np
from utility import TARGET_OUTPUTS, prepare_data
import time
import multiprocessing
import pickle

# Get the maximum number of available CPU cores
max_workers=multiprocessing.cpu_count()
print("max CPU threads ", max_workers)

# Function to retrieve parameters from the 'parameters.txt' file
def getParameters(filename='parameters.txt'):
    parameters = FileReader(filename).getParameters()
    return parameters

# Function to load a trained SOM model from a pickle file
def load_som_model(filename):
    with open(filename, 'rb') as file:
        som = pickle.load(file)
    return som

# Function to calculate the error between input data and the SOM
def calculate_error(data, som):
    error = 0
    for i in range(len(data)):
        som.setInputs(data[i])
        bmu = som.findBMU()
        error += sum((data[i][f'in{k}'] - som.inputNeurons[k - 1].getWeights()[bmu[0]][bmu[1]]) ** 2 for k in range(1, 17))
    error /= len(data)
    
    return error

# Function to train the SOM using the specified data file
def train(filename='all_data.txt'):
    # Get parameters from the 'parameters.txt' file
    parameters = getParameters()

     # Prepare data for training
    data = prepare_data(filename)
    print('Training Data Length:', len(data['trainingData']))
    try:
        # Attempt to load a pre-trained SOM model
        SOM = load_som_model(f"som_LR={parameters['learningRate']}_GS={parameters['gridSize']}_EP={parameters['numEpochs']}.pkl")
    except:
        # If loading fails, create a new SOM with specified parameters
        SOM = SOMNeuralNetwork(parameters)

    try:
        # Train the SOM over the specified number of epochs
       while SOM.currentEpoch < int(parameters["numEpochs"]):
            print(f'Epoch {SOM.currentEpoch + 1}')
            start_time = time.time()

            # Iterate through each pattern in the training data
            while SOM.currentPattern < len(data['trainingData']):
                print(f"Progress: {SOM.currentPattern}/{len(data['trainingData'])}", end="\r")
                
                # Update the SOM based on the current input pattern
                SOM.setInputs(data["trainingData"][SOM.currentPattern])
                bmu = SOM.findBMU()
                SOM.updateLearningRate()
                SOM.updateGaussianRadius()
                SOM.updateWeights(bmu)
                SOM.currentPattern += 1
            end_time = time.time()
            print(f'Epoch {SOM.currentEpoch + 1} completed in {end_time - start_time} seconds')
            
            # Calculate and append training error
            start_time = time.time()
            train_error = calculate_error(data['trainingData'], SOM)
            SOM.appendTrainingError(train_error)
            end_time = time.time()
            print(f'Epoch {SOM.currentEpoch+1} training error calculated in {end_time - start_time} seconds')
            print(f"Train Error={train_error}")

            # Calculate and append testing error
            start_time = time.time()
            test_error = calculate_error(data['testData'], SOM)
            SOM.appendTestingError(test_error)
            end_time = time.time()
            print(f'Epoch {SOM.currentEpoch+1} testing error calculated in {end_time - start_time} seconds')
            print(f"Test Error={test_error}")
            
            # Reset pattern count and move to the next epoch
            SOM.currentPattern = 0
            SOM.currentEpoch += 1
            
    except KeyboardInterrupt:
        # Handle KeyboardInterrupt (e.g., user interrupts training)
        print("Training interrupted")
    
    print("Training complete")

    # Save the trained SOM model to a pickle file
    SOM.save(f"som_LR={SOM.parameters['learningRate']}_GS={SOM.parameters['gridSize']}_EP={parameters['numEpochs']}.pkl")
   
    # Plot training and testing errors
    SOM.plotTrainingError()
    SOM.plotTestingError()
    
    # If letter mapping is not available, map letters and save the SOM model again
    if SOM.letterMap is None:
        SOM.mapLetters(data['testData'])
        SOM.save(f"som_LR={SOM.parameters['learningRate']}_GS={SOM.parameters['gridSize']}_EP={parameters['numEpochs']}.pkl")

    # Visualize the SOM map
    SOM.visualize_map()

# Set the random seed for reproducibility    
np.random.seed(2)  

# Call the train function to start the training process
train()

