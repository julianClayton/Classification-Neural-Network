import csv
import numpy as np



PATH = "data/uci_har_dataset/"

TESTING_DATA_FILE = "test/X_test.txt"
TESTING_LABELS_FILE = "test/y_test.txt"

TRAINING_DATA_FILE = "train/X_train.txt"
TRAINING_LABELS_FILE = "train/y_train.txt"


def read_data():
	print("Loading HAR training data...")
	training_X, training_y = extract_data(TRAINING_DATA_FILE, TRAINING_LABELS_FILE)
	print("...done")

	print("Loading HAR testing data...")
	testing_X, testing_y = extract_data(TESTING_DATA_FILE, TESTING_LABELS_FILE)

	print("...done")
	return training_X, training_y, testing_X, testing_y

def extract_data(f1, f2):
	data = []
	labels = []

	with open(PATH + f1) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=' ')
		for row in readCSV:
			features = []
			for i in range(len(row)):
				if row[i] != '':
					features.append(float(row[i]))
			data.append(features)

	with open(PATH + f2) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=' ')
		for row in readCSV:
			label = vectorize_label(int(row[0]))
			labels.append(label)

	return np.asarray(data), np.asarray(labels)

def vectorize_label(x):
	label = np.zeros(6)
	label[int(x-1)] = 1
	return label

if __name__ == "__main__":
	training_X, training_y, testing_X, testing_y = read_data()
	print(len(training_X))
