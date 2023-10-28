# import pandas as pd

# class PandasFileReader:
#     def __init__(self, filename):
#         self.filename = filename
#         self.df = pd.read_csv(self.filename, sep=' ')

#     def get_parameters(self):
#         parameters = self.df.set_index(0).squeeze().to_dict()
#         for key in parameters.keys():
#             try:
#                 parameters[key] = int(parameters[key])
#             except:
#                 try:
#                     parameters[key] = float(parameters[key])
#                 except:
#                     pass
#         return parameters

#     def get_data(self):
#         data = self.df.drop(0).set_index(self.df.columns[0]).T.to_dict('list')
#         return data


class FileReader:
    
    def __init__(self, filename):
        self.filename = filename
        self.file = open(self.filename, 'r')
        self.lines = self.file.readlines()
        self.file.close()

    def getLines(self):
        return self.lines
    
    def load_json_data(self):
        # read normalized_data.json file and organize it into a dictionary
        import json

        # Path to your JSON file
        json_file_path = self.filename

        # Step 1: Read the JSON file
        with open(json_file_path, 'r') as json_file:
            json_data = json_file.read()

        # Step 2: Parse the JSON string into a dictionary
        try:
            dictionary_data = json.loads(json_data)
            print("Successfully loaded JSON data into a dictionary.")
            return dictionary_data

        except json.JSONDecodeError as e:
            print("Error decoding JSON:", str(e))


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
    def getData(self,delimiter=' '):
        lines = self.lines
        data = {}

        # Extract keys from the first line
        keys = lines[0].split()
        for key in keys:
            data[key] = list()
        
        # Iterate through the rest of the lines
        for i in range(1, len(lines)):
            values = lines[i].split(delimiter)
            
            for col, value in enumerate(values):
                # Assign values to their respective keys
                try:
                    data[keys[col]].append(float(value))
                except:
                    data[keys[col]].append(value)


        return data
    
class FileWriter():
    def __init__(self, filename):
        self.filename = filename
    
    def write(self, labels, data, delimiter=' '):
        with open(self.filename, 'w+') as f:
            for label in labels:
                f.write(label + delimiter)
            f.write('\n')
            
            for record in data:
                for entry in record:
                    f.write(str(entry) + delimiter)
                f.write('\n')
            f.close()            
                
class Normalizer():

    def normalize(self):
        for key in self.data.keys():
            if key == 'out':
                self.normalizedData['out'] = self.data['out']
                continue

            self.normalizedData[key] = list()
            self.max[key] = max(self.data[key])
            self.min[key] = min(self.data[key])
            for value in self.data[key]:
                result = (value - self.min[key]) / (self.max[key] - self.min[key])
                # rounded_result = round(result, 2)
                self.normalizedData[key].append(result)

    def __init__(self, data):
        self.data = data
        self.normalizedData = {}
        self.max = {}
        self.min = {}
        self.normalize()


