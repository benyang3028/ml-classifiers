import scipy.io as sio
from scipy.spatial import distance
from random import sample
import numpy as np
from numpy import *
import matplotlib.pyplot as plt


def load_data():
    # load data
    file_path = "digits.mat"
    mat_contents = sio.loadmat(file_path)
    X_data, Y_data = mat_contents['X'], mat_contents['Y']

    return X_data, Y_data


def calculate_distance(X, Y, x_test, type):
    dist_list = np.reshape(distance.cdist(X, np.array([x_test]), type), (1, Y.size))[0]
    Y = np.reshape(Y, (1, Y.size))[0]
    distance_y_pair = list(zip(dist_list, Y))
    return distance_y_pair


def majority_class(k_nearest_distances):
    distance, y = zip(*k_nearest_distances)
    return np.bincount(y).argmax()


def find_knn_nearest_neighbor(X_train, Y_train, x_test, type='euclidean', k=1):
    distance_list = calculate_distance(X_train, Y_train, x_test, type)
    distance_list.sort(key=lambda x: x[0])
    k_nearest = distance_list[0:k]
    return majority_class(k_nearest)


def split_data(X, Y, population_size, sample_size):
    train_indices = sample(range(0, population_size), sample_size)
    test_indices = [x for x in range(0, 10000) if x not in train_indices]
    X_train, Y_train = X[train_indices, :], Y[train_indices, :]
    X_test, Y_test = X[test_indices, :], Y[test_indices, :]
    return X_train, Y_train, X_test, Y_test


def create_performance_graph(X, Y, population_size):
    for k in range(1, 6):
        test_size_accuracy_values = []
        for i in range(1000, 10000, 500):
            X_train, Y_train, X_test, Y_test = split_data(X, Y, population_size, i)
            test_size = Y_test.size
            y_values = []
            for j in range(0, test_size):
                y_pred = find_knn_nearest_neighbor(X_train, Y_train, X_test[j], k=k)
                y_values.append((y_pred, Y_test[j][0]))
            test_size_accuracy_values.append(accuracy(y_values, test_size))
        plt.plot(range(1000, 10000, 500), test_size_accuracy_values, label=f'k={k}')

    plt.title('k-NN Classifier Performance (k=3)')
    plt.legend(loc='upper right', fontsize='small')
    plt.xlabel('Training Sample Size')
    plt.ylabel('Accuracy')
    plt.savefig('k-NN Classifier Performance (Trial 1)')


def accuracy(y_values, test_size):
    correct = 0
    for y_pred, y_true in y_values:
        if y_pred == y_true:
            correct = correct + 1
    return correct / test_size


def single_classification(X, Y, sample_size, population_size, k=1):
    X_train, Y_train, X_test, Y_test = split_data(X, Y, population_size, sample_size)
    test_size = Y_test.size

    y_values = []
    for j in range(0, test_size):
        y_pred = find_knn_nearest_neighbor(X_train, Y_train, X_test[j], k=k)
        y_values.append((y_pred, Y_test[j][0]))
    return y_values, accuracy(y_values, test_size)


def main():
    X, Y = load_data()

    y_values, accuracy = single_classification(X, Y, 8000, 10000)
    print("k-NN Classifier of 2000 Test Samples over 8000 Training Samples, k=1")
    print("Predicted Y vs True Y for each Test Sample: ")
    print(y_values)
    print(f"Accuracy: {accuracy}")

    create_performance_graph(X, Y, 10000)


if __name__ == "__main__":
    main()
