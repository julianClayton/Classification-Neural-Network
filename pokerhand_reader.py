import csv
import numpy as np

PATH 			= "data/pokerhand/"
TRAINING_FILE 	= "pokerhand.data.training"
TESTING_FILE 	= "pokerhand.data.testing"

def read_data():
	print("loading pokerhand training data...")
	training_X, training_y	= extract_data(TRAINING_FILE)
	print("...done")
	print("loading pokerhand testing data...")
	testing_X, testing_y 	= extract_data(TESTING_FILE)
	print("...done")
	print("done loading.")
	return training_X, training_y, testing_X, testing_y

def extract_data(f):
		hands = []
		labels 	 = []
		with open(PATH + f) as csvfile:
			readCSV = csv.reader(csvfile, delimiter=',')
			numrows = 0
			for row in readCSV:
				features = np.zeros(10)
				for i in range(len(row)-1):
					features[i] = int(row[i])
				label = vectorize_label(row[len(row)-1])
				hands.append(features.tolist())
				labels.append(label.tolist())
		return np.asarray(hands), np.asarray(labels)

def vectorize_label(x): #One hot label
	label = np.zeros(10)
	label[int(x)] = 1
	return label

if __name__ == "__main__":
	read_data()