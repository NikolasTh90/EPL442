import concurrent.futures
import copyreg
import types
import multiprocessing
from functools import partial
import NeuralNetwork
from utility import TARGET_OUTPUTS
from time import time
def calculateDeltas(network:NeuralNetwork,outputs):
    for layer in reversed(network.getLayers()):
        if layer == network.getLayers()[-1]:
            for i, node in enumerate(layer):
                node.calculateDelta(targetOutput = outputs[i])
        else:
            if layer == network.getLayers()[0]:
                continue
            for node in layer:
                node.calculateDelta()





# def update_weights(node, learning_rate, momentum):
#     try:
#         node.updateWeights(learningRate=learning_rate, momentum=momentum)
#     except Exception as e:
#         pass

# def updateWeights(network):
#     learning_rate = network.getLearningRate()
#     momentum = network.getMomentum()


#     processes = []
#     for layer in network.getLayers():
#         for node in layer:
#             process = multiprocessing.Process(target=update_weights, args=(node, learning_rate, momentum))
#             processes.append(process)
#             process.start()

#     for process in processes:
#         process.join()








# def update_weights(node, learning_rate, momentum):
#     node.updateWeights(learningRate=learning_rate, momentum=momentum)

# def parallel_update_weights(node, learning_rate, momentum):
#     try:
#         update_weights(node, learning_rate, momentum)
#         return "Success"
#     except Exception as e:
#         return f"Failed: {str(e)}"

# def updateWeights(network):
#     learning_rate = network.getLearningRate()
#     momentum = network.getMomentum()

#     # # Ensure that Node.updateWeights is picklable
#     # copyreg.pickle(types.MethodType, lambda m: (getattr, (m.__func__, m.__self__)))

#     # # Create a partial function to handle pickling Node.updateWeights
#     # partial_update_weights = partial(parallel_update_weights, learning_rate=learning_rate, momentum=momentum)

#     with concurrent.futures.ProcessPoolExecutor() as executor:
        

#         for layer in network.getLayers():
#             futures = []
#             for node in layer:
#                 future = executor.submit(parallel_update_weights, node, learning_rate, momentum)
#                 futures.append(future)

#             # Wait for all tasks to complete
#             concurrent.futures.wait(futures)

#         # Check for any failures
#         for future in futures:
#             result = future.result()
#             if result != "Success":
#                 print("Error in weight update:", result)


def updateWeights(network:NeuralNetwork):
    for layer in network.getLayers():
        for node in layer:
            node.updateWeights(learningRate = network.getLearningRate(), momentum = network.getMomentum())


def backPropagation(network:NeuralNetwork, data:dict, patternNum:int):
    outputs = [1 if TARGET_OUTPUTS.index(data[patternNum]['out']) == i else 0 for i in range(len(TARGET_OUTPUTS))] 
    calculateDeltas(network, outputs)
    updateWeights(network)