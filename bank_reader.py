import csv
import numpy as np

PATH = 'data/bank/bank-additional/'
FILE = 'bank-additional-full.csv'
TRAINING_SIZE = 40000


jobs = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
marital_status = ['divorced', 'married', 'single', 'unknown']
education = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown']
default = ['no', 'yes', 'unknown']
housing_loan = ['no', 'yes', 'unknown']
personal_loan = ['no', 'yes', 'unknown']
contact = ['cellular', 'telephone']
month = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
day_of_week = ['mon', 'tue', 'wed', 'thu', 'fri']
poutcome = ['failure', 'nonexistent', 'success']

output = ['yes', 'no']

categories = [jobs, marital_status, education, default, housing_loan, personal_loan, contact, month, day_of_week, poutcome]

def read_data():
    print('loading Bank data-set...')
    data, labels = extract_data()
    print('...done')
    print('shuffling data...')
    perm = np.arange(len(data))
    np.random.shuffle(perm)
    data_shuffled = data[perm]
    labels_shuffled = labels[perm]
    training_X = data_shuffled[:TRAINING_SIZE]
    training_y = labels_shuffled[:TRAINING_SIZE]
    testing_X = data_shuffled[TRAINING_SIZE:]
    testing_y = labels_shuffled[TRAINING_SIZE:]
    return training_X, training_y, testing_X, testing_y


def category_to_num(cat_num, sub_cat):
    category = categories[cat_num]
    return category.index(sub_cat)


def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def extract_data():
    data = []
    labels = []
    with open(PATH + FILE) as (csvfile):
        readCSV = csv.reader(csvfile)
        for row in readCSV:
            features = []
            colcount = 0
            row = row[0].split(';')
            for i in range(len(row) - 1):
                c = eval(row[i])
                if is_float(c):
                    features.append(float(c))
                else:
                    features.append(category_to_num(colcount, c))
                    colcount += 1
            data.append(features)
            labels.append(vectorize_label(row[len(row) - 1]))
    return np.asarray(data), np.asarray(labels)


def vectorize_label(x):
    label = np.zeros(2)
    label[output.index(eval(x))] = 1
    return label


if __name__ == '__main__':
    read_data()