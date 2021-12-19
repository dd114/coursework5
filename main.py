# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import mnist_loader
from Network import Network


# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    # print(test_data[1][1])
    # print(len(test_data[0]))

    MNN = Network([784, 30, 10])
    MNN.SGD(training_data, 1, 10, 3.0, test_data = test_data)

    # digit in array for recognition
    index_digit_in_array = 3
    probability = MNN.feedforward(validation_data[index_digit_in_array][0])
    predict_digit = np.argmax(probability)
    precise_digit = validation_data[index_digit_in_array][1];
    print("Predict digit = {0}".format(predict_digit))
    print("Precise digit = {0} with a probability of = {1} %".format(precise_digit, int(probability[predict_digit][0] * 100)))

    plt.imshow(validation_data[index_digit_in_array][0].reshape(28, 28), cmap=plt.cm.binary)
    plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
