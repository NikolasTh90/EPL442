import numpy as np
from FileManager import DataHandler  # Assumed class for data handling
from NeuralNetwork import RBFNeuralNetwork  # Your RBF network implementation
import time

# Function to retrieve parameters from the 'parameters.txt' file
def getParameters(filename='parameters.txt'):
    parameters = {}  # Replace with actual code to read parameters
    return parameters

# Function to calculate the error for RBF Network
def calculate_error(network, data):
    total_error = 0
    for inputVector, target in data:
        outputs = network.forwardPass(inputVector)
        error = target - outputs
        total_error += np.sum(error ** 2)
    return total_error / len(data)

# Function to train the RBF Network using the specified data
def train(filename='all_data.txt'):
    # Get parameters from the 'parameters.txt' file
    parameters = getParameters()

    # Prepare data for training
    data_handler = DataHandler(filename)
    training_data, test_data = data_handler.get_data()  # Replace with actual code to get data

    # Initialize RBF Network with specified parameters
    RBFNetwork = RBFNeuralNetwork(parameters['numInputNeurons'], parameters['numRBFNeurons'], 
                                  parameters['numOutputNeurons'], parameters['centers'], 
                                  parameters['sigmas'])

    training_errors = []
    testing_errors = []

    # Train the RBF Network over the specified number of epochs
    for epoch in range(parameters['numEpochs']):
        print(f'Epoch {epoch + 1}')
        start_time = time.time()

        # Train the network with each input in the training data
        for inputVector, target in training_data:
            RBFNetwork.train(inputVector, target, parameters['learningRate'])

        # Calculate and store training error
        training_error = calculate_error(RBFNetwork, training_data)
        training_errors.append(training_error)

        # Calculate and store testing error
        testing_error = calculate_error(RBFNetwork, test_data)
        testing_errors.append(testing_error)

        end_time = time.time()
        print(f'Epoch {epoch + 1} completed in {end_time - start_time} seconds')
        print(f"Training Error: {training_error}, Testing Error: {testing_error}")

    # Plot training and testing errors
    plot_errors(training_errors, testing_errors)

def plot_errors(training_errors, testing_errors):
    import matplotlib.pyplot as plt
    plt.plot(training_errors, label='Training Error')
    plt.plot(testing_errors, label='Testing Error')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.title('RBF Neural Network Training and Testing Errors')
    plt.legend()
    plt.show()

# Set the random seed for reproducibility
np.random.seed(2)

# Call the train function to start the training process
train()