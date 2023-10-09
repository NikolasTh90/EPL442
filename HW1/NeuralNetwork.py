from . import Neuron
import numpy as np
class NeuralNetwork:
    def __init__(self, parameters:dict):
        self.parameters = parameters
        self.layers = []
        self.createLayers()
        self.connectLayers()
        self.initializeWeights()


    def createLayers(self):
        # Create the layers
        # Create the input layer  
        inputLayer = [Neuron(isBias=True)] 
        for i in range(int(self.parameters["numInputNeurons"])):
            inputLayer.append(Neuron())

        self.layers.append(inputLayer)

        # Create the hidden layers
        if self.parameters['numHiddenLayerOneNeurons'] > 0:
            # Create the first hidden layer
            hiddenLayerOne = [Neuron(isBias=True)]
            for i in range(int(self.parameters["numHiddenLayerOneNeurons"])):
                hiddenLayerOne.append(Neuron())
            
            self.layers.append(hiddenLayerOne)

        # Create the second hidden layer
        if self.parameters['numHiddenLayerTwoNeurons']:
            hiddenLayerTwo = [Neuron(isBias=True)]
            for i in range(int(self.parameters["numHiddenLayerTwoNeurons"])):
                hiddenLayerTwo.append(Neuron())
        
            self.layers.append(hiddenLayerTwo)

        # Create the output layer
        outputLayer = []
        for i in range(int(self.parameters["numOutputNeurons"])):
            outputLayer.append(Neuron())
        
        self.layers.append(outputLayer)

    def connectLayers(self):
        for i, layer in enumerate(self.layers):
            if i == len(self.layers - 1):
                break
            for node in layer:
                node.setConnectedToNeurons([next_layer_node for next_layer_node in self.layers[i+1]])
            for next_node in node.getConnectedToNeurons():
                next_node.addConnectedFromNeuron(node)

    def initializeWeights(self):
        for i, layer in enumerate(self.layers):
            if i == len(self.layers - 1):
                break
            for node in layer:
                node.setWeights(np.random.uniform(low=-1, high=1, size=len(self.layers[i+1])))
    
    def getLayers(self):
        return self.layers

    
