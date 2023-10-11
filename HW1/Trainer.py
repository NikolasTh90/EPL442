from MyFileReader import FileReader
from NeuralNetwork import NeuralNetwork
from FeedForward import feedForward
from BackPropagation import backPropagation
def getData():
    trainingData = FileReader('training.txt').getData()
    testData = FileReader('test.txt').getData()
    return {"trainingData": trainingData,
             "testData": testData}

def getParameters():
    parameters = FileReader('parameters.txt').getParameters()
    return parameters

def train(epochs = 1000):
    parameters = getParameters()
    data = getData()
    neuralNetwork = NeuralNetwork(parameters)
    print(data)
    for patternNum in range(len(data['testData']['in1'])):
            print(feedForward(neuralNetwork, data['testData'], patternNum))
    for epoch in range(epochs):
        for patternNum in range(len(data['trainingData']['in1'])):
            feedForward(neuralNetwork, data['trainingData'], patternNum)
            backPropagation(neuralNetwork, data['trainingData'], patternNum)
            print(neuralNetwork.getLayers()[0][1].getWeights())
            # print(neuralNetwork.getLayers()[0][1].getConnectedToNeurons())
    print("Training complete")

    for patternNum in range(len(data['testData']['in1'])):
            print(feedForward(neuralNetwork, data['testData'], patternNum))
    
train(3)

