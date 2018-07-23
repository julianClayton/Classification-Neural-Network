"""Functions for downloading and reading MNIST data.
some code used and changed from: https://github.com/tensorflow/models/blob/master/official/mnist/dataset.py

"""
import gzip
import os
import numpy

PATH = "data/mnist/"
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
VALIDATION_SIZE = 5000

def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number')
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename):
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number')
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        return dense_to_one_hot(labels)


def process_images(images, labels):
    assert images.shape[0] == labels.shape[0], ("images.shape: %s labels.shape: %s" % (images.shape,labels.shape))
    assert images.shape[3] == 1
    images = images.reshape(images.shape[0],images.shape[1] * images.shape[2])
    images = images.astype(numpy.float32)
    images = numpy.multiply(images, 1.0 / 255.0)
    return images, labels

def read_data_sets():
    print("loading mnist data....")
    local_file = os.path.join(PATH, TRAIN_IMAGES)
    train_images = extract_images(local_file)

    local_file = os.path.join(PATH, TRAIN_LABELS)
    train_labels = extract_labels(local_file)

    local_file = os.path.join(PATH, TEST_IMAGES)
    test_images = extract_images(local_file)

    local_file = os.path.join(PATH, TEST_LABELS)
    test_labels = extract_labels(local_file)

    validation_images = train_images[:VALIDATION_SIZE]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_images = train_images[VALIDATION_SIZE:]
    train_labels = train_labels[VALIDATION_SIZE:]
    training_X, training_y  = process_images(train_images, train_labels)
    testing_X, testing_y    = process_images(validation_images, validation_labels)
    print("...done")
    return training_X, training_y, testing_X, testing_y