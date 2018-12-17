import csv


# remember to start row_number from 1, index 0 is for column labels,
# maximum row_number for this set of data is 1599

def get_row(row_number):
    with open('../data/winequality-red.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != row_number or line_count == 0:
                line_count += 1
                continue
            else:
                return InputData(row[0], row[1], row[2], row[3], row[4], row[5], row[6],
                                 row[7], row[8], row[9], row[10], row[11])


class InputData:

    def __init__(self, fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide,
                 total_sulfur_dioxide, density, pH, sulphates, alcohol, quality):
        self.fixed_acidity = fixed_acidity
        self.volatile_acidity = volatile_acidity
        self.citric_acid = citric_acid
        self.residual_sugar = residual_sugar
        self.chlorides = chlorides
        self.free_sulfur_dioxide = free_sulfur_dioxide
        self.total_sulfur_dioxide = total_sulfur_dioxide
        self.density = density
        self.pH = pH
        self.sulphates = sulphates
        self.alcohol = alcohol
        self.quality = quality
