import scipy.io as sio
from scipy.stats import multivariate_normal
from random import sample
import numpy as np
from numpy import *
import matplotlib.pyplot as plt


def load_data(threshold):

    digits = "digits.mat"
    m = sio.loadmat(digits)
    X = m['X']
    Y = m['Y'] 

    covariance = np.cov(X, rowvar=False, bias=True)
    n = np.shape(covariance)[0]
    idx = [i for i in range(0,n) if covariance[i, i]>threshold]
    X = X[:, idx]
    return X, Y


def filter(X, Y, digit):
    i = np.where(Y == [digit])[0]
    X_filter = X[i]
    Y_filter = Y[i]
    return X_filter, Y_filter


def class_prior(Y, n):
    return math.log(Y.size / n)


def train_data(X):
    covariance = np.cov(X, rowvar=False, bias=True)
    mean = np.mean(X, axis=0)
    mvnorm = multivariate_normal(mean=mean, cov=covariance)
    return mvnorm


def P_classifier(trained_data, x):
    p = []
    for i in range(0, 10):
        eta_x = trained_data.get(i)[0].logpdf(x)
        probability = eta_x + trained_data.get(i)[1]
        p.append(probability)
    return np.argmax(p)

def single_classification(X, Y, sample_size, pop_size):
    X_train, Y_train, X_test, Y_test = split_data(X, Y, sample_size, pop_size)
    test_size = Y_test.size
    y_values = []

    trained_data = {}
    for num in range(0, 10):
        X_filter, Y_filter = filter(X_train, Y_train, num)
        mvnorm = train_data(X_filter)
        trained_data[num] = [mvnorm, class_prior(Y_filter, 10000)]

    for i in range(0, test_size):
        prediction = P_classifier(trained_data, X_test[i])
        y_values.append((prediction, Y_test[i][0]))

    return y_values, accuracy(y_values, test_size)

def split_data(X, Y, sample_size, pop_size):
    train = sample(range(0, pop_size), sample_size)
    test = []
    for i in range(0, 10000):
        if i not in train:
            test.append(i)

    X_train = X[train, :]
    Y_train = Y[train, :]
    X_test = X[test, :]
    Y_test = Y[test, :]
    return X_train, Y_train, X_test, Y_test


def create_performance_graph(X, Y, pop_size):
    accuracy_values = []
    for i in range(5000, 10000, 50):
        X_train, Y_train, X_test, Y_test = split_data(X, Y, i, pop_size)
        test_size = Y_test.size
        y_values = []

        trained_data = {}
        for num in range(0, 10):
            X_filter = filter(X_train, Y_train, num)[0]
            Y_filter = filter(X_train, Y_train, num)[1]
            mvnorm = train_data(X_filter)
            trained_data[num] = (mvnorm, class_prior(Y_filter, 10000))

        for i in range(0, test_size):
            prediction = P_classifier(trained_data, X_test[i])
            y_values.append((prediction, Y_test[i][0]))
        accuracy_values.append(accuracy(y_values, test_size))

    plt.plot(range(5000, 10000, 50), accuracy_values)
    plt.title('Performance of Probablistic Classifier')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Accuracy')


def accuracy(y, size):
    count = 0
    for i, j in y:
        if i == j:
            count = count + 1
    return count / size


def main():

    X, Y = load_data(12000)
    
    y_values, accuracy = single_classification(X, Y, 8000, 10000)
    print("Probabilistic Classification of 2000 Test Samples over 8000 Training Samples")
    print("Predicted Y vs True Y for each Test Sample: ")
    print(y_values)
    print(f"Accuracy: {accuracy}")

    create_performance_graph(X, Y, 10000)


if __name__ == "__main__":
    main()
