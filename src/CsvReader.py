import csv
import copy


# remember to start row_number from 1, index 0 is for column labels,
# maximum row_number for this set of data is 1599

# returns an array containing 12 float numbers describing one wine, last item of an array is
# an output
def get_row(row_number, filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != row_number or line_count == 0:
                line_count += 1
                continue
            else:
                # return InputData(row[0], row[1], row[2], row[3], row[4], row[5], row[6],
                #                  row[7], row[8], row[9], row[10], row[11])
                return [float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
                    , float(row[5]), float(row[6]), float(row[7]), float(row[8]),
                        float(row[9]), float(row[10]), float(row[11])]


# returns an array of rows , each row is an array of float values.
def get_data(last_row, filename):
    data = []
    for i in range(1, last_row):
        data.append(get_row(i, filename))

    return data


# you need to pass two existing arrays : data and outputs.
# Outputs should be an empty array, and data should be an array
# created by get_data function. As result this function modifies data array and outputs array.
# Data array will contain only inputs and outputs array will contain only outputs.
# Outputs is one-dimensional array.
def seperate_inputs_and_outputs(data, outputs):
    for i in range(0, len(data)):
        outputs.append([data[i].pop(11)])


# this function finds 11 maximal values, for each wine's feature.
# Returns an array of found maximal values. They will be needed for normalizing data.
def find_max(data):
    max = 0.0
    maxes = []
    for i in range(0, 11):
        for j in range(0, len(data)):
            if (data[j][i] > max):
                max = data[j][i]
        maxes.append(max)
        max = 0.0

    maxes.append(10)  # this is the output node, it is within 0 to 10 rate
    return maxes


# This function normalizes every feature of wine.
# The result is that every feature is from 0 - 1 .
# This function modifies given data array
def normalize_data(data):
    maxes = find_max(data)

    for i in range(0, 12):
        for j in range(0, len(data)):
            data[j][i] /= maxes[i]
        if maxes[i] == 0:
            data[j][i] = maxes[i]


def get_normalized_data():
    data = get_data(1600, '../data/winequality-red.csv')
    normalize_data(data)
    return data


def get_training_data(rowNumber, data):
    trainingData = copy.deepcopy(data)

    for i in range(1598, rowNumber-1, -1):
        del trainingData[i]

    return trainingData


def get_testing_data(rowNumber, data):
    testingData = copy.deepcopy(data)

    for i in range(0, rowNumber):
        del testingData[0]

    return testingData
