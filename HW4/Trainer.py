import numpy as np
from FileManager import DataHandler, FileReader, FileWriter  # Assumed class for data handling
from NeuralNetwork import RBFNeuralNetwork  # Your RBF network implementation
import time



# Function to calculate the error for RBF Network
def calculate_error(network, data):
    total_error = 0
    target = data['Activity'].to_numpy() # Assuming 'Activity' column exists
    inputVector = data.drop(['Activity', 'Compound'], axis=1).to_numpy() # Assuming 'Compound' column exists
    for inputVector, target in zip(inputVector, target):
        outputs = network.forwardPass(inputVector)
        error = target - outputs
        total_error += np.sum(error ** 2)
    return total_error / len(data)

# Function to train the RBF Network using the specified data
def train(parameters, centers, training_data, test_data):
    inputVector = training_data.drop(['Activity', 'Compound'], axis=1).to_numpy()
    # Initialize RBF Network with specified parameters
    RBFNetwork = RBFNeuralNetwork(parameters['numInputNeurons'], parameters['numHiddenLayerNeurons'], 
                                  parameters['numOutputNeurons'], centers,
                                  parameters['sigmas'])

    training_errors = []
    testing_errors = []

    # Train the RBF Network over the specified number of epochs
    for epoch in range(1, parameters['maxIterations']+1):
        print(f'Epoch {epoch}')
        start_time = time.time()

        # Train the network with each input in the training data
        target = training_data['Activity'].to_numpy() # Assuming 'Activity' column exists
        inputVector = training_data.drop(['Activity', 'Compound'], axis=1).to_numpy() # Assuming 'Compound' column exists
        RBFNetwork.train(inputVector, target, parameters['learningRates'])


        # Calculate and store training error
        training_error = calculate_error(RBFNetwork, training_data)
        training_errors.append(training_error)

        # Calculate and store testing error
        testing_error = calculate_error(RBFNetwork, test_data)
        testing_errors.append(testing_error)

        end_time = time.time()
        print(f'Epoch {epoch} completed in {end_time - start_time} seconds')
        print(f"Training Error: {training_error}, Testing Error: {testing_error}")
    
    # Write the errors to a file
    error_log_file = FileWriter(f"results_LR={parameters['learningRates']}_S={parameters['sigmas']}_EP={parameters['maxIterations']}.txt")
    error_log_file.write(['epoch', 'trainingError', 'testError'], zip(range(1, epoch+ 1), training_errors , testing_errors))

    # write the weights to a file
    weights_log_file = FileWriter(f"weights_LR={parameters['learningRates']}_S={parameters['sigmas']}_EP={parameters['maxIterations']}.txt")
    weights_log_file.write_weights(RBFNetwork.rbfLayer, RBFNetwork.outputLayer)

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
np.random.seed(5)
# Prepare data for training
# Get parameters from the 'parameters.txt' file
parameters = FileReader('parameters.txt').getParameters()   
try:
    # If training and testing files are provided, use them
    train_filename = parameters['trainFile'] if parameters['trainFile'] else 'training.txt'
    test_filename= parameters['testFile'] if parameters['testFile'] else 'test.txt'
    data_handler = DataHandler(train_filename)
    train_data = data_handler.read_data()
    data_handler = DataHandler(test_filename)
    test_data = data_handler.read_data()   

except FileNotFoundError:
    print('Training or Testing File not found')
    time.sleep(1)
    if parameters['dataFile']:
        print('Attempting to create them by splitting the full data file')
        data_handler = DataHandler(parameters['dataFile'] if parameters['dataFile'] else 'selwood.txt')
        df = data_handler.read_data()
        df_preprocessed = data_handler.preprocess_data(df)
        train_data, test_data = data_handler.split_data(df_preprocessed, 10)
        # Saving the data to files
        data_handler.save_data(train_data, parameters['trainFile'] if parameters['trainFile'] else 'training.txt')
        data_handler.save_data(test_data, parameters['testFile'] if parameters['testFile'] else 'test.txt')
        print('Files created successfully')
        time.sleep(1)

# Read the centers from file
try:
    centers_filename = parameters['centresFile'] if parameters['centresFile'] else 'centers.txt'
    data_handler = DataHandler(centers_filename)
    centers = data_handler.read_data()
    centers = centers.to_numpy()
except FileNotFoundError:
    print('Centers file not found')
    time.sleep(1)
    if parameters['centresFile']:
        print('Attempting to create it by clustering the data')
        centers = data_handler.cluster_data(train_data, n_clusters=parameters['numHiddenLayerNeurons'])
        # Saving the data to files
        data_handler.save_data(centers, parameters['centresFile'] if parameters['centresFile'] else 'centers.txt')
        print('File created successfully')
        time.sleep(1)
# centers.to_numpy()
    
# Call the train function to start the training process
train(parameters, centers, train_data, test_data)