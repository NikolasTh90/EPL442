import json

TARGET_OUTPUTS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                  'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                    'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def normalize_data(filename='all_data.txt', delimiter=','):
    from FileManager import FileWriter, FileReader, Normalizer
    data = FileReader(filename).getData(delimiter)
    return Normalizer(data).normalizedData
    


def load_json_data(filename='normalized_data.json'):
    from FileManager import FileReader
    return FileReader(filename).load_json_data()

def sort_data(data):
    sorted_data = {}
    for letter in TARGET_OUTPUTS:
        sorted_data[letter] = {}
        for key in data.keys():
            sorted_data[letter][key] = list()
    for i in range(len(data['out'])):
        sorted_data[data['out'][i]]['out'].append(data['out'][i])
        for key in data.keys():
            if key == 'out':
                continue
            sorted_data[data['out'][i]][key].append(data[key][i])
    return sorted_data


def split_data(data, training_percentage=0.75):
    training_data = {}
    test_data = {}
    
    for letter in data:
        # print(letter)
        training_data[letter] = {}
        test_data[letter] = {}
        for counter, _ in enumerate(data[letter]['out']):
            for key in data[letter].keys():

                if counter <= training_percentage * len(data[letter]['out']):
                    try: 
                        training_data[letter][key].append(data[letter][key][counter])
                    except:
                        training_data[letter][key] = list()
                        training_data[letter][key].append(data[letter][key][counter])

                else:
                    try: 
                        test_data[letter][key].append(data[letter][key][counter])
                    except:
                        test_data[letter][key] = list()
                        test_data[letter][key].append(data[letter][key][counter])

    return {'training_data': training_data, 'test_data': test_data}

def shuffle_data(data):
    import random
    random.shuffle(data)
    return data

def organize_data(data):
    # Step 1: Reorganize the data into a list of dictionaries
    # print(data)
    reorganized_data = []
    for key, values in data.items():
        # print('key', key)
        # print("values", values)
        try:
            for i in range(len(values['out'])):
                
                entry = {'out': values['out'][i]}
                for j in range(1, 17):  # in1 to in16
                    entry[f'in{j}'] = values[f'in{j}'][i]
                reorganized_data.append(entry)
        except:
            continue
            
    
    print('ORGANIZED\n')
    
    # Step 2: Shuffle the data
    reorganized_data = shuffle_data(reorganized_data)
    print('SHUFFLED\n')
    return reorganized_data
    # # Step 3: Create a new dictionary with the specified structure and populate with shuffled data
    # new_dictionary = {'out': [], 'in1': [], 'in2': [], 'in3': [], 'in4': [], 'in5': [],
    #                 'in6': [], 'in7': [], 'in8': [], 'in9': [], 'in10': [], 'in11': [],
    #                 'in12': [], 'in13': [], 'in14': [], 'in15': [], 'in16': []}

    # for entry in reorganized_data:
    #     new_dictionary['out'].append(entry['out'])
    #     for j in range(1, 17):
    #         new_dictionary[f'in{j}'].append(entry[f'in{j}'])
    
    # print('NEW\n', new_dictionary)
    # return new_dictionary


def prepare_data(filename='all_data.txt'):
    data = normalize_data(filename)
    data = sort_data(data)
    data = split_data(data)
    train_data = organize_data(data['training_data'])
    test_data = organize_data(data['test_data'])
    return {'trainingData': train_data, 'testData': test_data}
    





