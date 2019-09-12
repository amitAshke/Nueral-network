import numpy
import sys

import Trainer


def shuffle(train_x, train_y):
    randomize = numpy.arange(len(train_x))
    numpy.random.shuffle(randomize)
    train_x = train_x[randomize]
    train_y = train_y[randomize]

    train_x = numpy.asarray(train_x)
    train_y = numpy.asarray(train_y)
    return train_x, train_y

def load_data_with_validation(train_data, train_class, test_data):
    train_x = numpy.loadtxt(train_data)
    train_y = numpy.loadtxt(train_class)
    test_x = numpy.loadtxt(test_data)

    # numpy.save("saved_train_data", train_x)
    # numpy.save("saved_train_class", train_y)
    # numpy.save("saved_test_data", test_x)

    train_x = numpy.load("saved_train_data.npy")
    train_y = numpy.load("saved_train_class.npy")
    test_x = numpy.load("saved_test_data.npy")

    train_x, train_y = shuffle(train_x, train_y)

    validation_size = int(train_x.shape[0] * 0.2)
    validation_x, validation_y = train_x[-validation_size:, :], train_y[-validation_size:]
    train_x, train_y = train_x[:-validation_size, :], train_y[:-validation_size]
    train_x = train_x / 255.0
    validation_x = validation_x / 255.0
    test_x = test_x / 255.0
    return train_x, train_y, validation_x, validation_y, test_x

def load_data_without_validation(train_data, train_class, test_data):
    train_x = numpy.loadtxt(train_data)
    train_y = numpy.loadtxt(train_class)
    test_x = numpy.loadtxt(test_data)
    train_x, train_y = shuffle(train_x, train_y)
    return train_x, train_y, test_x


if __name__ == '__main__' :
    # train_x, train_y, validation_x, validation_y, test_x = load_data_with_validation("train_x", "train_y", "test_x")

    train_x, train_y, test_x = load_data_without_validation("train_x", "train_y", "test_x")

    trainer = Trainer.Trainer(train_x, train_y)

    trainer.train_without_validation()

    # trainer.train_with_validation(validation_x, validation_y)

    trainer.write_test_y(test_x)
