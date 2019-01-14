from CsvReader import *
from Network import Network, save_network_to_file, load_network_from_file, get_rating

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import time

wines = get_normalized_data('../data/winequality-red.csv')
poor_wines = get_poor_wines(wines)
good_wines = get_good_wines(wines)
average_wines = get_average_wines(wines)

# poor_wines_outputs = []
# poor_wines_inputs = []
# seperate_inputs_and_outputs(poor_wines, poor_wines_inputs, poor_wines_outputs)

# good_wines_outputs = []
# good_wines_inputs = []
# seperate_inputs_and_outputs(good_wines, good_wines_inputs, good_wines_outputs)

# average_wines_outputs = []
# average_wines_inputs = []
# seperate_inputs_and_outputs(average_wines, average_wines_inputs, average_wines_outputs)


elements = 500

training_input_set = []
training_output_set = []
training_set = []

for data in get_training_data(int(elements / 21), poor_wines):
    for i in range(0, 21):
        training_set.append(copy.deepcopy(data))
for data in get_training_data(int(elements / 6), good_wines):
    for i in range(0, 6):
        training_set.append(copy.deepcopy(data))
for data in get_training_data(elements, average_wines):
    training_set.append(copy.deepcopy(data))

# print(training_set)
seperate_inputs_and_outputs(training_set, training_input_set, training_output_set)

testing_input_set = []
testing_output_set = []
testing_set = []


for data in get_testing_data(int(elements / 21), len(poor_wines), poor_wines):
    # for i in range(0, 21):
        testing_set.append(copy.deepcopy(data))
for data in get_testing_data(int(elements / 6) , len(good_wines), good_wines):
    # for i in range(0, 6):
        testing_set.append(copy.deepcopy(data))
for data in get_testing_data(elements, len(average_wines), average_wines):
    testing_set.append(copy.deepcopy(data))

seperate_inputs_and_outputs(testing_set, testing_input_set, testing_output_set)


network = Network()
network.learningRate = 0.2

n_epoch = 100
times = []
ratios = []
epochs = []
for i in range(10 , n_epoch, 10):
        start = time.time()
        epochs.append(i)
        for j in range(0 , i):
                loss_sum = 0
                for k in range(0, len(training_input_set)):
                        result = network.feed_forward(training_input_set[k])
                        network.backward_propagation(training_output_set[k][0], result, training_input_set[k])
                        loss_sum += abs(result - training_output_set[k][0])

        wrong = 0
        correct = 0
        # print("\n===TESTING===\n")
        for l in range(0, len(testing_input_set)):
                result = network.feed_forward(testing_input_set[l]) \
                        # if round(result, 1) == testing_outputs[i][0]:
                if get_rating(result) == get_rating(testing_output_set[l][0]):
                        # print("ROW %d - CORRECT" % i)
                        # print(result, testing_output_set[i][0])
                        correct += 1
                else:
                        # print("ROW %d - WRONG!!! \n %f %f" % (i, result, testing_output_set[i][0]))
                        # print(result, testing_output_set[i][0])
                        wrong += 1

        end = time.time()
        times.append(end - start)
        ratios.append(correct / len(testing_input_set) * 100)
        #print("CORRECT: %d WRONG: %d  ratio = %f" % (correct, wrong, correct / len(testing_input_set) * 100))

plt.plot( epochs, ratios , linestyle="-", marker='o')
plt.xlabel("epochs")
plt.ylabel("ratio")
plt.savefig('../diagrams/testplot.png')