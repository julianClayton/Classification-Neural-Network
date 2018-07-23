import csv
import numpy as np


PATH = "data/wine_quality/"
RED_WINE_FILE = "winequality-red.csv"
WHITE_WINE_FILE = "winequality-white.csv"
TRAINING_SIZE = 5000


def read_data():
	redwine_data, redwine_labels 		= extract_data(RED_WINE_FILE)
	whitewine_data, whitewine_labels 	= extract_data(WHITE_WINE_FILE)
	all_data = np.concatenate((redwine_data, whitewine_data))
	all_labels = np.concatenate((redwine_labels, whitewine_labels))
	perm = np.arange(len(all_data))
	np.random.shuffle(perm)
	data_shuffled = all_data[perm]
	labels_shuffled = all_labels[perm]

	training_X = data_shuffled[:TRAINING_SIZE]
	training_y = labels_shuffled[:TRAINING_SIZE]

	testing_X = data_shuffled[TRAINING_SIZE:]
	testing_y = labels_shuffled[TRAINING_SIZE:]

	return  training_X, training_y,testing_X, testing_y



def extract_data(f):
	data = []
	labels = []

	with open(PATH + f) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=';')
		for row in readCSV:
			features = []
			for i in range(len(row)-1):
				features.append(float(row[i]))
			label = vectorize_label(row[len(row)-1])
			labels.append(label.tolist())
			data.append(features)
			#print(label)

	return np.asarray(data), np.asarray(labels)

#for this dataset it says 0-10 but there are no 0s,1s,2s or 10s so I just did it 0-9
def vectorize_label(x): #One hot label
	label = np.zeros(10)
	label[int(x)] = 1
	return label


if __name__ == "__main__":
	read_data()