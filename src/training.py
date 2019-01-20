from CsvReader import *
from Network import Network, save_network_to_file, load_network_from_file, get_rating
from random import shuffle

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import time


wines = get_normalized_data('../data/winequality-red.csv')
poor_wines = get_poor_wines(wines)      #only wines with quality less than 6.5
good_wines = get_good_wines(wines)      #only wines with quality greater than 6.5

training_size = 4/5                    #fraction of wines being a training set

training_input_set = []
training_output_set = []
training_set = []

nr_of_poor_wines = int(training_size * len(poor_wines))         #nr of training bad wines is a fraction of whole set of poor wines
nr_of_good_wines = int(training_size * len(good_wines))         #nr of training good wines is a fraction of whole set of good wines
copies_of_good_wines = int(nr_of_poor_wines / nr_of_good_wines) #amount of copies of good wines so the amount of good and poor wines is the same

print("Good = " + str(nr_of_good_wines))
print("Poor = " + str(nr_of_poor_wines))
print("Copies = " + str(copies_of_good_wines))

for data in get_training_data(nr_of_poor_wines, poor_wines):
        training_set.append(copy.deepcopy(data))
for data in get_training_data(nr_of_good_wines, good_wines):
    for i in range(0, copies_of_good_wines):                    #here we clone good wine few times.
        training_set.append(copy.deepcopy(data))

shuffle(training_set)

seperate_inputs_and_outputs(training_set, training_input_set, training_output_set)

print(len(training_input_set))
print(len(training_output_set))
print(len(training_set))

testing_input_set = []
testing_output_set = []
testing_set = []

for data in get_testing_data(nr_of_poor_wines, len(poor_wines), poor_wines):    #we take remaining wines to be a testing set
        testing_set.append(copy.deepcopy(data))
for data in get_testing_data(nr_of_good_wines , len(good_wines), good_wines):
    for i in range(0, copies_of_good_wines):
        testing_set.append(copy.deepcopy(data))


seperate_inputs_and_outputs(testing_set, testing_input_set, testing_output_set)
print(len(testing_input_set))
print(len(testing_output_set))
print(len(testing_set))


network = Network()
network.learningRate = 0.05

copy_network = copy.deepcopy(network)

n_epoch = 400

step = 20

def train(lr):
        network = copy.deepcopy(copy_network)
        network.learningRate = lr

        times = []
        ratios = []
        epochs = []
        for i in range(0 , n_epoch, step):
                start = time.time()

                

                for j in range(0 , step):
                        loss_sum = 0
                        for k in range(0, len(training_input_set)):
                                result = network.feed_forward(training_input_set[k])
                                network.backward_propagation(training_output_set[k][0], result, training_input_set[k])
                                loss_sum += abs(result - training_output_set[k][0])

                wrong = 0
                correct = 0
                for l in range(0, len(testing_input_set)):
                        result = network.feed_forward(testing_input_set[l]) 
                        if get_rating(result) == get_rating(testing_output_set[l][0]):
                                correct += 1
                        else:
                                wrong += 1

                end = time.time()
                times.append(end - start)
                epochs.append(i)
                ratios.append(correct / len(testing_input_set) * 100)
                print(i)

        plt.plot( epochs, ratios , linestyle="-", marker='o')
        plt.xlabel("epochs")
        plt.ylabel("ratio")
        plt.suptitle('train-set ' + str(training_size) + " lr=0.08" , fontSize=12)

        plt.savefig('../diagrams/loweringLR3.png')

lr = 0.08
train(lr)

