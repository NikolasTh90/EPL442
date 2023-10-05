class NeuralNetwork:

    def __init__(self, layers:dict, learningRate, momentum, maxIterations, trainFile, testFile):
        self.layers = layers
        self.learningRate = learningRate
        self.momentum = momentum
        self.maxIterations = maxIterations
        self.trainFile = trainFile
        self.testFile = testFile

    def train(self):
            
