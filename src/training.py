from CsvReader import *
from Network import Network, save_network_to_file, load_network_from_file, get_rating
from random import shuffle

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import time

def train_network():
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


    elements = 40

    training_input_set = []
    training_output_set = []
    training_set = []

    # for data in get_training_data(int(elements / 21), poor_wines):
    #     for i in range(0, 21):
    #         training_set.append(copy.deepcopy(data))
    # for data in get_training_data(int(elements / 6), good_wines):
    #     for i in range(0, 6):
    #         training_set.append(copy.deepcopy(data))
    # for data in get_training_data(elements, average_wines):
    #     training_set.append(copy.deepcopy(data))

    for data in get_training_data(40, poor_wines):
            training_set.append(data)
    for data in get_training_data(40, good_wines):
            training_set.append(data)
    for data in get_training_data(40, average_wines):
            training_set.append(data)

    shuffle(training_set)

    # print(training_set)
    seperate_inputs_and_outputs(training_set, training_input_set, training_output_set)

    print(len(training_input_set))
    print(len(training_output_set))
    print(len(training_set))
    # normalize_output(training_output_set)

    testing_input_set = []
    testing_output_set = []
    testing_set = []

    for data in get_testing_data(40, len(poor_wines), poor_wines):
        # for i in range(0, 21):
            testing_set.append(data)
    for data in get_testing_data(40 , len(poor_wines), good_wines):
        # for i in range(0, 6):
            testing_set.append(data)
    for data in get_testing_data(40, len(poor_wines), average_wines):
            testing_set.append(data)



    # for data in get_testing_data(int(elements / 21), len(poor_wines), poor_wines):
    #     for i in range(0, 21):
    #         testing_set.append(copy.deepcopy(data))
    # for data in get_testing_data(int(elements / 6) , len(good_wines), good_wines):
    #     for i in range(0, 6):
    #         testing_set.append(copy.deepcopy(data))
    # for data in get_testing_data(elements, len(average_wines), average_wines):
    #     testing_set.append(copy.deepcopy(data))

    seperate_inputs_and_outputs(testing_set, testing_input_set, testing_output_set)
    print(len(testing_input_set))
    print(len(testing_output_set))
    print(len(testing_set))
    #normalize_output(testing_output_set)


    network = Network()
    network.learningRate = 0.2

    n_epoch = 5000
    times = []
    ratios = []
    epochs = []
    step = 100



    for i in range(0 , n_epoch, step):
            start = time.time()

            # if(i >= 1300):
            #         network.learningRate = 0.1
            # elif( i >= 1500):
            #         network.learningRate = 0.05

            if i >= 1200:
                    network.learningRate = 0.1
            elif i>= 2000:
                    network.learningRate = 0.05
            elif i>=3000:
                    network.learningRate = 0.001

            if( i % 200 == 0 ):
                    shuffle(training_set)
                    training_input_set = []
                    training_output_set = []

                    # print(training_set)
                    seperate_inputs_and_outputs(training_set, training_input_set, training_output_set)

            for j in range(0 , step):
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
            epochs.append(i)
            ratios.append(correct / len(testing_input_set) * 100)
            
            #print("CORRECT: %d WRONG: %d  ratio = %f" % (correct, wrong, correct / len(testing_input_set) * 100))

    plt.plot( epochs, ratios , linestyle="-", marker='o')
    plt.xlabel("epochs")
    plt.ylabel("ratio")
    plt.suptitle('red wine 40 elements of each wine lr=0.01 to 0.005->300e no norm', fontSize=12)

    # plt.suptitle('learning rate = ' + repr(network.learningRate) + ' , ' + repr(elements) + ' elements of every wine in training set', fontSize=12)
    # plt.figtext(0.99, 0.01, "learning rate = 0.2 , 40 elements of every wine in training set", horizontalalignment='right')
    # plt.annotate("learning rate = 0.2 , 40 elements of every wine in training set", (0,0), (0, -40), xycoords='axes fraction', textcoords='offset points', va='top')

    plt.savefig('../diagrams/noNormred' + repr(elements) +'testplot' + repr(network.learningRate) + '.png')
    plt.show()
    return network

def test_network_automatically(network):
    wrong = 0
    correct = 0
    whole_data = get_normalized_data('winequality-red.csv')
    edge_row = 1599
    testing_input_set = get_testing_data(edge_row, whole_data)
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
    print("CORRECT: %d WRONG: %d  ratio = %f" % (correct, wrong, correct / len(testing_inputs) * 100))

def test_network_manually(network):
    fixed_acidity = input("fixed_acidity = ")
    volatile_acidity = input("volatile_acidity = ")
    citric_acid = input("citric acid = ")
    residual_sugar = input("residual sugar = ")
    chlorides = input("chlorides = ")
    free_sulfur_dioxide = input("free sulfur dioxide = ")
    total_sulfur_dioxide = input("total sulfur dioxide = ")
    density = input("density = ")
    pH = input("pH = ")
    sulphates = input("sulphates = ")
    alcohol = input("alcohol = ")

    result = network.feed_forward([float(fixed_acidity), float(volatile_acidity), float(citric_acid), float(residual_sugar), float(chlorides), float(free_sulfur_dioxide), float(total_sulfur_dioxide), float(density), float(pH), float(sulphates), float(alcohol)])
    print(get_rating(result))