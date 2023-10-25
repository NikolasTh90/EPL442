from FileManager import FileReader, FileWriter
from NeuralNetwork import NeuralNetwork
from FeedForward import feedForward
from BackPropagation import backPropagation
import math
from matplotlib import pyplot as plt
import numpy as np
from utility import TARGET_OUTPUTS, prepare_data
import time
import multiprocessing
import copy
import os
max_workers=multiprocessing.cpu_count()
print(max_workers)
def getParameters(filename='parameters.txt'):
    parameters = FileReader('parameters.txt').getParameters()
    return parameters

# Define a function to calculate pattern error
def calculate_pattern_error(actual_output, target_output):
    pattern_error = sum(0.5 * (target - actual) ** 2 for actual, target in zip(actual_output, target_output))
    return pattern_error

# Modify the function to calculate training accuracy
def calculate_accuracy(neural_network, data, pattern_num):
    actual_output = feedForward(neural_network, data, pattern_num)
    target_output = [1 if TARGET_OUTPUTS.index(data[pattern_num]['out']) == i else 0 for i in range(len(TARGET_OUTPUTS))]
    
    if TARGET_OUTPUTS[actual_output.index(max(actual_output))] == data[pattern_num]['out']:
        success = 1
    else:
        success = 0
    
    pattern_error = calculate_pattern_error(actual_output, target_output)
    return pattern_error, success


# Define a function to process a pattern and calculate accuracy
def process_pattern_wrapper(args):
    pattern_num, neural_network, data = args
    # print(f'Process {multiprocessing.current_process().pid} processing pattern {pattern_num}')
    return calculate_accuracy(neural_network[int(multiprocessing.current_process().pid)%max_workers], data, pattern_num)

# Define a function to calculate training accuracy in parallel
def calculate_accuracy_parallel(neural_network, data):
    errors = []
    successes = []

    with multiprocessing.Pool(processes=max_workers) as pool:
        # Create a list of deep copies of the neural network
        neural_network_copies = [copy.deepcopy(neural_network) for _ in range(max_workers)]
        
        # Use process_pattern_wrapper to pass arguments, assigning each worker its copy
        results = pool.map(process_pattern_wrapper, [(pattern_num, neural_network_copies, data) for i, pattern_num in enumerate(range(len(data)))])

    # Extract errors and successes from the results
    for pattern_error, success in results:
        errors.append(pattern_error / 26)
        successes.append(success)

    # Calculate overall training error and success rate
    total_error = sum(errors) / len(data)
    total_success_rate = (sum(successes) / len(data)) * 100

    return total_error, total_success_rate

def train(filename='all_data.txt', epochs=None):
    parameters = getParameters()
    data = prepare_data(filename)
    print('Training Data Length:', len(data['trainingData']))
    print('Test Data Length:', len(data['testData']))
    neuralNetwork = NeuralNetwork(parameters)
    if not epochs:
        epochs = parameters['maxIterations']
    trainingError = list()
    trainingSuccess = list()
    testError = list()
    testSuccess = list()
    try:
        for epoch in range(epochs):
            beforeEpoch = time.time()

            for patternNum in range(len(data['trainingData'])): # changed after reorganizing data
                # before = time.time()
                feedForward(neuralNetwork, data['trainingData'], patternNum)
                # after = time.time()
                # print(f'Feedforward {patternNum} completed in {after-before} seconds')
                # before = time.time()
                backPropagation(neuralNetwork, data['trainingData'], patternNum)
                # after = time.time()
                # print(f'Backpropagation {patternNum} completed in {after-before} seconds')
            afterEpoch = time.time()
            print(f'Epoch {epoch+1} completed in {afterEpoch-beforeEpoch} seconds')
    
            beforeTrainingAccuracy = time.time()
            # Training accuracy
            # error = 0
            # success = 0
            # for patternNum in range(len(data['trainingData'])): # changed after reorganizing data
            #     actualOutput = feedForward(neuralNetwork, data['trainingData'], patternNum)
            #     if TARGET_OUTPUTS[actualOutput.index(max(actualOutput))] == data['trainingData'][patternNum]['out']:
            #         success = success + 1
            #     outputs = [1 if TARGET_OUTPUTS.index(data['trainingData'][patternNum]['out']) == i else 0 for i in range(len(TARGET_OUTPUTS))] 
            #     patternError = 0
            #     for i, output in enumerate(outputs):
            #         tempError = 0.5 * math.pow(( output - actualOutput[i]), 2)
            #         patternError = patternError + tempError
            #     error = error + (patternError/len(actualOutput))
            # trainingError.append(error/len(data['trainingData']))
            # trainingSuccess.append( 100 * success / float(len(data['trainingData'])))

            errors, successes = calculate_accuracy_parallel(neuralNetwork, data['trainingData'])
            trainingError.append(errors)
            trainingSuccess.append(successes)
            afterTrainingAccuracy = time.time()
            print(f'Training accuracy calculated in {afterTrainingAccuracy-beforeTrainingAccuracy} seconds, error: {errors}, success: {successes} ')

            beforeTestAccuracy = time.time()
            # Test accuracy
            # error = 0
            # success = 0
            # for patternNum in range(len(data['testData'])): # changed after reorganizing data
            #     actualOutput = feedForward(neuralNetwork, data['testData'], patternNum)
            #     if TARGET_OUTPUTS[actualOutput.index(max(actualOutput))] == data['testData'][patternNum]['out']:
            #         success = success + 1
            #     outputs = [1 if TARGET_OUTPUTS.index(data['testData'][patternNum]['out']) == i else 0 for i in range(len(TARGET_OUTPUTS))] 
            #     patternError = 0
            #     for i, output in enumerate(outputs):
            #         tempError = 0.5 * math.pow(( output - actualOutput[i]), 2)
            #         patternError = patternError + tempError
            #     error = error + (patternError/len(actualOutput))
            # testError.append(error/len(data['testData']))
            # testSuccess.append( 100 * success / float(len(data['testData'])))
            errors, successes = calculate_accuracy_parallel(neuralNetwork, data['testData'])
            testError.append(errors)
            testSuccess.append(successes)
            afterTestAccuracy = time.time()
            print(f'Test accuracy calculated in {afterTestAccuracy-beforeTestAccuracy} seconds , error: {errors}, success: {successes} ')
    except KeyboardInterrupt:
        print("Training interrupted")

    


    print("Training complete")


    error_log_file = FileWriter(f"error  (LR={neuralNetwork.getLearningRate()}, M={neuralNetwork.getMomentum()}, L1={parameters['numHiddenLayerOneNeurons']}, L2={parameters['numHiddenLayerTwoNeurons']}).txt")
    error_log_file.write(['epoch', 'trainingError', 'testError'], zip(range(1, epochs + 1), trainingError , testError))
    success_log_file = FileWriter(f"success  (LR={neuralNetwork.getLearningRate()}, M={neuralNetwork.getMomentum()}, L1={parameters['numHiddenLayerOneNeurons']}, L2={parameters['numHiddenLayerTwoNeurons']}).txt")
    success_log_file.write(['epoch', 'trainingSuccess', 'testSuccess'], zip(range(1, epochs + 1), trainingSuccess, testSuccess))
   

    # Plotting success rates
    plt.figure(f"TrainSuccess-TestSuccess Rate (LR={neuralNetwork.getLearningRate()}, M={neuralNetwork.getMomentum()})")
    plt.plot(trainingSuccess, label=f'Success Training Rate')
    plt.plot(testSuccess, label='Success Test Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.show()

    # Plotting error rates
    plt.figure("Train-Test Error (LR={neuralNetwork.getLearningRate()}, M={neuralNetwork.getMomentum()})")
    plt.plot(trainingError, label='Error Training Rate')
    plt.plot(testError, label='Error Test Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.legend()
    plt.show()

    
np.random.seed(4)    
train()

