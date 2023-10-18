from FileManager import FileReader, FileWriter
from NeuralNetwork import NeuralNetwork
from FeedForward import feedForward
from BackPropagation import backPropagation
import math
from matplotlib import pyplot as plt
import numpy as np


def getData():
    trainingData = FileReader('training.txt').getData()
    testData = FileReader('test.txt').getData()
    return {"trainingData": trainingData,
             "testData": testData}

def getParameters():
    parameters = FileReader('parameters.txt').getParameters()
    return parameters

def train(epochs=None):
    parameters = getParameters()
    data = getData()
    neuralNetwork = NeuralNetwork(parameters)
    print(data)
    if not epochs:
        epochs = parameters['maxIterations']
    trainingError = list()
    trainingSuccess = list()
    testError = list()
    testSuccess = list()
    for epoch in range(epochs):
        for patternNum in range(len(data['trainingData']['out'])):
            feedForward(neuralNetwork, data['trainingData'], patternNum)
            backPropagation(neuralNetwork, data['trainingData'], patternNum)
        
    # Training accuracy
        error = 0
        success = 0
        successThreshold = 0.25
        for patternNum in range(len(data['trainingData']['out'])):
            if math.fabs(feedForward(neuralNetwork, data['trainingData'], patternNum) - data['trainingData']['out'][patternNum]) <= successThreshold:
                success = success + 1
            tempError = 0.5 * math.pow(( data['trainingData']['out'][patternNum] - feedForward(neuralNetwork, data['trainingData'], patternNum)), 2)
            error = error + tempError
        trainingError.append(error)
        trainingSuccess.append( 100 * success / float(len(data['trainingData']['out'])))

        # Test accuracy
        error = 0
        success = 0
        for patternNum in range(len(data['testData']['out'])):
            if math.fabs(feedForward(neuralNetwork, data['testData'], patternNum) - data['testData']['out'][patternNum]) <= successThreshold:
                success = success + 1
            tempError = 0.5 * math.pow(( data['testData']['out'][patternNum] - feedForward(neuralNetwork, data['testData'], patternNum)), 2)
            error = error + tempError
        testError.append(error)
        testSuccess.append(100 * success / float(len(data['testData']['out']))  )
    


    print("Training complete")


    error_log_file = FileWriter('errors.txt')
    error_log_file.write(['epoch', 'trainingError', 'testError'], zip(range(1, epochs + 1), trainingError , testError))
    success_log_file = FileWriter('success.txt')
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

    
np.random.seed(2)    
train()

