# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import mnist_loader
from Network import Network


# Third-party libraries
import numpy as np
# import matplotlib.pyplot as plt


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    # print(test_data[1][1])
    # print(len(test_data[0]))

    MNN = Network([784, 30, 10])
    MNN.SGD(training_data, 1, 10, 3.0, test_data = test_data)

    print(np.argmax(MNN.feedforward(test_data[0][0])))
    print(test_data[0][1])


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
