import pandas as pd

class PandasFileReader:
    def __init__(self, filename):
        self.filename = filename
        self.df = pd.read_csv(self.filename, sep=' ')

    def get_parameters(self):
        parameters = self.df.set_index(0).squeeze().to_dict()
        for key in parameters.keys():
            try:
                parameters[key] = int(parameters[key])
            except:
                try:
                    parameters[key] = float(parameters[key])
                except:
                    pass
        return parameters

    def get_data(self):
        data = self.df.drop(0).set_index(self.df.columns[0]).T.to_dict('list')
        return data


class FileReader:
    
    def __init__(self, filename):
        self.filename = filename
        self.file = open(self.filename, 'r')
        self.lines = self.file.readlines()
        self.file.close()

    def getLines(self):
        return self.lines


    # read the file paramters of this format and load in a parameters dictionary
    # numHiddenLayerOneNeurons 2
    # numHiddenLayerTwoNeurons 0
    # numInputNeurons 2
    # numOutputNeurons 1
    # learningRate 0.3
    # momentum 0.2
    # maxIterations 200
    # trainFile training.txt
    # testFile test.txt 
    def getParameters(self):
        parameters = {}
        for line in self.lines:
            line = line.strip()
            if line:
                line = line.split()
                parameters[line[0]] = line[1]
        for key in parameters.keys():
            try:
                parameters[key] = int(parameters[key])
            except:
                try:
                    parameters[key] = float(parameters[key])
                except:
                    pass
        return parameters
    
    # read the file parameters of this format and load in a parameters dictionary
    # input1 input2 output
    # 0 0 0
    # 0 1 1
    def getData(self):
        lines = self.lines
        data = {}

        # Extract keys from the first line
        keys = lines[0].split()
        for key in keys:
            data[key] = list()
        
        # Iterate through the rest of the lines
        for i in range(1, len(lines)):
            values = lines[i].split()
            
            for col, value in enumerate(values):
                # Assign values to their respective keys
                try:
                    data[keys[col]].append(float(value))
                except:
                    data[keys[col]].append(value)


        return data