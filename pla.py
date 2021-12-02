import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from matplotlib import cm
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D

class Perceptron(object):
    def __init__(self):
        self.lr = 0.2
        self.num_it = 30
        self.weights = None
        self.bias = None
    
    def activation_function(self, x):
        return 1 if x>= 0 else -1
    
    def prediction(self, X):
        activation = np.dot(X, self.weights) + self.bias
        predicted = self.activation_function(activation)
        return predicted
    
    def fit(self, X, Y, filename):
        self.weights = np.zeros(X.shape[1])
        self.bias = 0
        y = np.array([1 if item > 0 else -1 for item in Y])
        outfile = open(filename, 'w')
        for _ in range(self.num_it):
            for i, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)
                dw = self.lr * (y[i] - y_predicted)
                self.bias += dw
                self.weights += dw * x_i
            outfile.write("{0:.2f},{1:.2f},{2:.2f}\n".format(self.weights[0], self.weights[1], self.bias))
        outfile.close()

def main():
    args = sys.argv
    args.pop(0)
    data_file = args[0]
    df = pd.read_csv(data_file, header=None)

    X = np.array([(i[0], i[1]) for i in df.values])
    Y = np.array([i[2] for i in df.values])
    perceptron = Perceptron()
    perceptron.fit(X, Y, args[1])

    # visualization
    # fig = plt.figure()
    # axes = fig.add_subplot(1, 1, 1)
    # X1, X2 = X[:, 0], X[:, 1]
    # plt.scatter(X1, X2, marker="o", c=Y)
    
    # x1_min = np.amin(X[:, 0])
    # x1_max = np.amax(X[:, 0])
    # x2_1 = (-1 * perceptron.weights[0] * x1_min - perceptron.bias) / perceptron.weights[1]
    # x2_2 = (-1 * perceptron.weights[0] * x1_max - perceptron.bias) / perceptron.weights[1]

    # axes.plot([x1_min, x1_max], [x2_1, x2_2], "k")
    # plt.show()


if __name__ == "__main__":
    """DO NOT MODIFY"""
    main()