import MyFileReader as MyFileReader
parameters = MyFileReader.FileReader('parameters.txt').getParameters()
print(type(parameters), parameters)

