import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from matplotlib import cm
import matplotlib.lines as mlines
import statistics
from mpl_toolkits.mplot3d import Axes3D
import plot_db

def hypothesis(betas, X):
    n = X.shape[1]
    h = np.ones(X.shape[0])
    for i in range(X.shape[0]):
        h[i] = float(betas @ X[i])
    return h

def get_cost(X, y, betas):
    m = y.size
    sme = np.dot(X, betas.T) - y
    cost = (1/m) * 0.5 * sum(np.square(hypothesis(betas, X) - y))
    return cost, sme

def gradient_descent(betas, X, y, alpha):
    costs = np.ones(100)
    m = y.size
    for i in range(100):
        cost, sme = get_cost(X, y, betas)
        predictions = hypothesis(betas, X)
        err = predictions - y
        sum_delta = (alpha / m) * X.transpose().dot(err)
        betas = betas - sum_delta
        costs[i] = cost
    return betas, costs

def linear_regression(X, y, alpha):
    n = X.shape[1]
    betas = np.zeros(n)
    betas, cost = gradient_descent(betas, X, y, alpha)
    return betas, cost


def main():
    """
    YOUR CODE GOES HERE
    Implement Linear Regression using Gradient Descent, with varying alpha values and numbers of iterations.
    Write to an output csv file the outcome betas for each (alpha, iteration #) setting.
    Please run the file as follows: python3 lr.py data2.csv, results2.csv
    """
    args = sys.argv
    args.pop(0)
    
    data_file = args[0]
    out_file = open(args[1], 'w')
    data = pd.read_csv(data_file, header=None)
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]

    x1 = np.array([data[0][i] for i in range(len(data.values))])
    x2 = np.array([data[1][i] for i in range(len(data.values))])
    y = np.array([data[2][i] for i in range(len(data.values))])
    
    x1_scaled = np.array([(x-x1.mean())/x1.std() for x in x1])
    x2_scaled = np.array([(x-x2.mean())/x2.std() for x in x2])

    X = np.column_stack((x1_scaled, x2_scaled))
    
    one_column = np.ones((X.shape[0],1))
    X = np.concatenate((one_column, X), axis = 1)

    data = {
        'age': x1,
        'weight': x2,
        'height' : y
    }

    for alpha in alphas:
        betas, cost = linear_regression(X, y, alpha)
        cost = list(cost)
        n_iter = [i for i in range(0, 100)]
        out_file.write("{}, {}, {}, {}, {}\n".format(alpha, len(n_iter), betas[0], betas[1], betas[2]))

    betas, cost = linear_regression(X, y, 0.6)
    out_file.write("{}, {}, {}, {}, {}\n".format(0.6, len(n_iter), betas[0], betas[1], betas[2]))
    # plots of cost vs number of iterations

    # plt.plot(n_iter, cost)
    # plt.xlabel('No. of iterations')
    # plt.ylabel('Cost')
    # plt.show()

    predicted_vals = hypothesis(betas, X)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x1, x2, predicted_vals)
    ax.scatter(x1, x2, y)

if __name__ == "__main__":
    main()