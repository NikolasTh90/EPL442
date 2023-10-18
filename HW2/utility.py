import json

TARGET_OUPUTS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                  'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                    'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def normalize_data(filename='all_data.txt', delimiter=','):
    from FileManager import FileWriter, FileReader, Normalizer
    data = FileReader(filename).getData(delimiter)
    return Normalizer(data).normalizedData
    


def load_json_data(filename='normalized_data.json'):
    from FileManager import FileReader
    return FileReader(filename).load_json_data()

def organize_data(data):
    organized_data = {}
    for letter in TARGET_OUPUTS:
        organized_data[letter] = {}
        for key in data.keys():
            organized_data[letter][key] = list()
    for i in range(len(data['out'])):
        organized_data[data['out'][i]]['out'].append(data['out'][i])
        for key in data.keys():
            if key == 'out':
                continue
            organized_data[data['out'][i]][key].append(data[key][i])
    return organized_data

json.dump(normalize_data('data_sample.txt'), open('normalized_data.json', 'w+'))
# print(load_json_data())
training_data = {}
test_data = {}
data = organize_data(load_json_data())
for letter in data:
    print(letter)
    training_data[letter] = {}
    test_data[letter] = {}
    for counter, _ in enumerate(data[letter]['out']):
        for key in data[letter].keys():
            if key == 'out':
                continue
            if counter <= 0.75 * len(data[letter]):
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

print(training_data)


