from Neuron import Neuron
import copy
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
        if self.parameters['numHiddenLayerTwoNeurons'] > 0:
            hiddenLayerTwo = [Neuron(isBias=True)]
            for i in range(int(self.parameters["numHiddenLayerTwoNeurons"])):
                hiddenLayerTwo.append(Neuron())
        
            self.layers.append(hiddenLayerTwo)

        # Create the output layer
        outputLayer = []
        for i in range(int(self.parameters["numOutputNeurons"])):
            outputLayer.append(Neuron(isOutput=True))
        
        self.layers.append(outputLayer)

    def connectLayers(self):
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                break
            for node in layer:
                node.setConnectedToNeurons([next_layer_node  for next_layer_node in self.layers[i+1] if not next_layer_node.isBias ])
                for next_node in node.getConnectedToNeurons():
                    if not next_node.isBias:
                        next_node.addConnectedFromNeuron(node)

    def initializeWeights(self):
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                break
            for node in layer:
                node.setWeights(np.random.uniform(low=-1, high=1, size=len(node.getConnectedToNeurons())))
                node.weightsOld = copy.copy(node.getWeights())
        
    
    def getLayers(self):
        return self.layers
    
    def getLearningRate(self):
        return float(self.parameters['learningRate'])

    def getMomentum(self):
        return float(self.parameters['momentum'])
    
    

    def __str__(self):
        nodes = ""
        for i, layer in enumerate(self.layers):
            nodes += f"\n -- {i} --\n "
            for node in layer:
                nodes += f"\n {node}\n"

            
            
        return "Neural Network: \n" + \
            "Parameters: " + str(self.parameters) + "\n" + \
            "Layers: " + "\n" + nodes
            
