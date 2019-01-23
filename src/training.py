from src.CsvReader import *
from src.Network import Network, get_rating
from random import shuffle
from src.NetworkCupy import NetworkCupy

import matplotlib
import matplotlib.pyplot as plt
import time

matplotlib.use('TkAgg')


def test_network_automatically(network):
    wines = get_normalized_data('../data/winequality-red.csv')
    poor_wines = get_poor_wines(wines)  # only wines with quality less than 6.5
    good_wines = get_good_wines(wines)  # only wines with quality greater than 6.5

    training_size = 4 / 5  # fraction of wines being a training set

    nr_of_poor_wines = int(
        training_size * len(poor_wines))  # nr of training bad wines is a fraction of whole set of poor wines
    nr_of_good_wines = int(
        training_size * len(good_wines))  # nr of training good wines is a fraction of whole set of good wines
    copies_of_good_wines = int(
        nr_of_poor_wines / nr_of_good_wines)  # amount of copies of good wines so the amount of good and poor wines is the same

    testing_input_set = []
    testing_output_set = []
    testing_set = []

    for data in get_testing_data(nr_of_poor_wines, len(poor_wines),
                                 poor_wines):  # we take remaining wines to be a testing set
        testing_set.append(copy.deepcopy(data))
    for data in get_testing_data(nr_of_good_wines, len(good_wines), good_wines):
        for i in range(0, copies_of_good_wines):
            testing_set.append(copy.deepcopy(data))

    seperate_inputs_and_outputs(testing_set, testing_input_set, testing_output_set)

    wrong = 0
    correct = 0
    for l in range(0, len(testing_input_set)):
        result = network.feed_forward(testing_input_set[l])
        if get_rating(result) == get_rating(testing_output_set[l][0]):
            correct += 1
        else:
            wrong += 1

    print("Tested " + str(len(testing_input_set)) + " datasets.")
    print("Result: " + str(correct / len(testing_input_set) * 100) + " % of good predictions")


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

    row = [float(fixed_acidity), float(volatile_acidity), float(citric_acid), float(residual_sugar), float(chlorides),
           float(free_sulfur_dioxide), float(total_sulfur_dioxide), float(density), float(pH), float(sulphates),
           float(alcohol)]

    data = get_whole_data('../data/winequality-red.csv')
    normalize_row(row, data)

    result = network.feed_forward(row)
    print(get_rating(result))


def train_network(use_cupy, lr, n_epoch, filename):
    wines = get_normalized_data('../data/winequality-red.csv')
    poor_wines = get_poor_wines(wines)  # only wines with quality less than 6.5
    good_wines = get_good_wines(wines)  # only wines with quality greater than 6.5

    training_size = 4 / 5  # fraction of wines being a training set

    training_input_set = []
    training_output_set = []
    training_set = []

    nr_of_poor_wines = int(
        training_size * len(poor_wines))  # nr of training bad wines is a fraction of whole set of poor wines
    nr_of_good_wines = int(
        training_size * len(good_wines))  # nr of training good wines is a fraction of whole set of good wines
    copies_of_good_wines = int(
        nr_of_poor_wines / nr_of_good_wines)  # amount of copies of good wines so the amount of good and poor wines is the same

    for data in get_training_data(nr_of_poor_wines, poor_wines):
        training_set.append(copy.deepcopy(data))
    for data in get_training_data(nr_of_good_wines, good_wines):
        for i in range(0, copies_of_good_wines):  # here we clone good wine few times.
            training_set.append(copy.deepcopy(data))

    shuffle(training_set)

    seperate_inputs_and_outputs(training_set, training_input_set, training_output_set)

    testing_input_set = []
    testing_output_set = []
    testing_set = []

    for data in get_testing_data(nr_of_poor_wines, len(poor_wines),
                                 poor_wines):  # we take remaining wines to be a testing set
        testing_set.append(copy.deepcopy(data))
    for data in get_testing_data(nr_of_good_wines, len(good_wines), good_wines):
        for i in range(0, copies_of_good_wines):
            testing_set.append(copy.deepcopy(data))

    seperate_inputs_and_outputs(testing_set, testing_input_set, testing_output_set)

    if not use_cupy:
        network = Network()
    else:
        network = NetworkCupy()

    copy_network = copy.deepcopy(network)

    step = 20

    network = copy.deepcopy(copy_network)
    network.learningRate = lr

    times = []
    ratios = []
    epochs = []
    for i in range(0, n_epoch, step):
        start = time.time()

        for j in range(0, step):
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

    plt.plot(epochs, ratios, linestyle="-", marker='o')
    plt.xlabel("epochs")
    plt.ylabel("ratio")
    plt.suptitle('train-set ' + str(training_size) + ' learning rate: ' + str(lr), fontSize=12)

    plt.savefig('../diagrams/' + str(filename) + '.png')
    return network
