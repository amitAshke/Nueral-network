import numpy
import matplotlib.pyplot as plt

class Trainer :

    def __init__(self, train_x, train_y, classes=10, hiddenLayerSize=125, eta=0.02, epochs=25):
        self.train_x = train_x
        self.train_y = train_y
        self.hiddenLayerSize = hiddenLayerSize
        self.eta = eta
        self.epochs = epochs
        self.weights1 = numpy.random.uniform(-0.05, 0.05, (hiddenLayerSize, 28 * 28))
        self.weights2 = numpy.random.uniform(-0.05, 0.05, (classes, hiddenLayerSize))
        self.bias1 = numpy.random.uniform(-0.05, 0.05, (hiddenLayerSize,))
        self.bias2 = numpy.random.uniform(-0.05, 0.05, (classes,))

    def train_with_validation(self, validation_x, validation_y):
        original_eta = self.eta
        learning_decay = self.eta / self.epochs

        training_loss = numpy.zeros(self.epochs)
        validation_loss = numpy.zeros(self.epochs)
        validation_probability = numpy.zeros(self.epochs)

        self.min_max_normalization(self.train_x)

        for epoch in range(self.epochs):
            total_loss = 0.0

            self.train_x, self.train_y = self.shuffle(self.train_x, self.train_y)

            for x, y in zip(self.train_x, self.train_y) :
                prediction, fp_params = self.front_propagation(x)

                loss = self.calculate_loss(prediction, y)
                total_loss += loss

                bp_params = self.back_propagation(x, y, fp_params)

                self.update_params(bp_params)

            training_loss_avg = total_loss / self.train_x.shape[0]
            validation_loss_avg, validation_correct = self.validation_calculate_loss(validation_x, validation_y)

            print ("epoch: {}:".format(epoch + 1))
            print ("train loss: {} | validation loss: {} | validation accuracy: {}".format(training_loss_avg, validation_loss_avg,
                                                                                    validation_correct * 100))
            training_loss[epoch] = training_loss_avg
            validation_loss[epoch] = validation_loss_avg
            validation_probability[epoch] = validation_correct;

            self.eta -= learning_decay

        # fig1, ax1 = plt.subplots()
        # ax1.plot(range(self.epochs), training_loss, label='training loss')
        # ax1.plot(range(self.epochs), validation_loss, label='validation loss')
        # ax1.legend(loc='upper right', shadow=True, fontsize='x-large')
        #
        # fig1, ax1 = plt.subplots()
        # ax1.plot(range(self.epochs), validation_probability, label='success probability')
        # ax1.legend(loc='upper left', shadow=True, fontsize='x-large')
        #
        # plt.show()

        self.eta = original_eta

        return

    def train_without_validation(self):
        original_eta = self.eta
        learning_decay = self.eta / self.epochs

        self.min_max_normalization(self.train_x)

        for epoch in range(self.epochs):
            total_loss = 0.0

            self.train_x, self.train_y = self.shuffle(self.train_x, self.train_y)

            for x, y in zip(self.train_x, self.train_y) :
                prediction, fp_params = self.front_propagation(x)

                loss = self.calculate_loss(prediction, y)
                total_loss += loss

                # go back to calc gradients
                bp_params = self.back_propagation(x, y, fp_params)

                # update the params
                self.update_params(bp_params)

            # training_loss_avg = total_loss / self.train_x.shape[0]
            #
            # print("epoch: {}:".format(epoch + 1))
            # print("train loss: {}".format(training_loss_avg))

            self.eta -= learning_decay

        self.eta = original_eta

        return

    def min_max_normalization(self, data_array):
        for column in range(len(data_array[0])):
            min_arg = float(min(data_array[:, column]))
            max_arg = float(max(data_array[:, column]))

            for row in range(len(data_array)):
                if min_arg == max_arg:
                    data_array[row, column] = 0
                else:
                    data_array[row, column] = (float(data_array[row, column]) - min_arg) / (max_arg - min_arg)

    def shuffle(self, train_x, train_y):
        randomize = numpy.arange(len(train_x))
        numpy.random.shuffle(randomize)
        train_x = train_x[randomize]
        train_y = train_y[randomize]

        train_x = numpy.asarray(train_x)
        train_y = numpy.asarray(train_y)
        return train_x, train_y

    def front_propagation(self, x):
        z1 = numpy.dot(self.weights1, x) + self.bias1
        # z1 = numpy.dot(self.weights1, x) / self.weights1.shape[1] + self.bias1
        h1 = self.ReLU(z1)
        z2 = numpy.dot(self.weights2, h1) + self.bias2
        # z2 = numpy.dot(self.weights2, h1) / self.weights2.shape[1] + self.bias2
        h2 = self.softmax(z2)
        return h2, [h1, h2, z1, z2]

    def back_propagation(self, x, y, fp_params):
        h1, h2, z1, z2 = [fp_params[index] for index in range(4)]

        # calculate the gradients
        z2_derivative = h2.dot(self.weights2) - self.weights2[int(y), :]
        h1_derivative = self.ReLU_derivative(z1)
        bias1_derivative = z2_derivative * h1_derivative
        weights1_derivative = numpy.outer(bias1_derivative, x)

        weights2_derivative = numpy.outer(h2, h1)
        weights2_derivative[int(y)] -= h1
        bias2_derivative = numpy.copy(h2)
        bias2_derivative[int(y)] -= 1
        return [weights1_derivative, weights2_derivative, bias1_derivative, bias2_derivative]

    def sigmoid(self, x):
        return 1 / (1 + numpy.exp(-x))

    def ReLU(self, x):
        return numpy.maximum(0, x)

    def softmax(self, x):
        exps = numpy.exp(x - x.max())
        return exps / exps.sum()

    def calculate_loss(self, prediction, y):
        return -numpy.log(prediction[int(y)])

    def sigmoid_derivative(self, x):
        sigx = self.sigmoid(x)
        return sigx * (1 - sigx)

    def ReLU_derivative(self, x):
        return (x > 0).astype(numpy.float)

    def update_params(self, bp_params):
        self.weights1 -= self.eta * bp_params[0]
        self.weights2 -= self.eta * bp_params[1]
        self.bias1 -= self.eta * bp_params[2]
        self.bias2 -= self.eta * bp_params[3]

    def validation_calculate_loss(self, validation_x, validation_y):
        loss_sum = 0.0
        correct = 0.0

        for x, y in zip(validation_x, validation_y):
            prediction, fp_params = self.front_propagation(x)

            loss = self.calculate_loss(prediction, y)
            loss_sum += loss

            if numpy.argmax(prediction) == y:
                correct += 1
        return loss_sum / validation_x.shape[0], correct / validation_x.shape[0]

    def write_test_y(self, test_x):
        with open("test_y", "w") as test_y:
            for x in test_x:
                y_hat = self.predict(x)
                test_y.write(str(y_hat))
                test_y.write("\n")

    def predict(self, x):
        z1 = numpy.dot(self.weights1, x) + self.bias1
        h1 = self.ReLU(z1)
        z2 = numpy.dot(self.weights2, h1) + self.bias2
        h2 = self.softmax(z2)
        return numpy.argmax(h2)