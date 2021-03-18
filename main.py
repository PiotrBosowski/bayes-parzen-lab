import statistics
from math import log

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import scipy.io as sio

from confusion import Confusion


class ParzenEstimator:
    def __init__(self, X_train):
        self.n = len(X_train)
        self.dim = len(X_train.columns)
        self.X_train = X_train

    def phi(self, row):
        for item in row:
            if item > 0.5:
                return False
        return True

    def count_within_cube(self, row, var, h):
        diff = pd.DataFrame()
        diff[var] = self.X_train[var] - row[var]
        diff[var] /= h
        within_cube = len(diff[(diff[var] < 0.5) & (diff[var] > -0.5)])
        return within_cube

    def pdf(self, row, var, h):
        volume = h ** self.dim
        return (1 / (self.n * volume)) * self.count_within_cube(row, var, h)


class ParzenBayesClassifier:
    def __init__(self):
        self.prior_probabilities = {}
        self.variable_distr_A1 = {}
        self.variable_distr_A2 = {}
        self.variables = []

    def fit(self, X_train, y_train):
        y_col = y_train.columns[-1]
        self.variables = X_train.columns
        self.prior_probabilities = dict.fromkeys(y_train[y_col].unique())
        for cls in self.prior_probabilities.keys():
            count = len(y_train[y_train[y_col] == cls])
            self.prior_probabilities[cls] = count / len(y_train)
        self.variable_distr_A1 = dict.fromkeys(X_train.columns)
        self.variable_distr_A2 = dict.fromkeys(X_train.columns)
        train_A1 = X_train[y_train[y_col] == 0]
        train_A2 = X_train[y_train[y_col] == 1]
        for var in X_train.columns:
            self.variable_distr_A1[var] = ParzenEstimator(train_A1)
            self.variable_distr_A2[var] = ParzenEstimator(train_A2)

    def calculate_bayes(self, row, size=0.6):  # 100% acc for size=0.5
        for var in self.variables:
            row['output_A1'] += log(
                1 + self.variable_distr_A1[var].pdf(row, var, size))
            row['output_A2'] += log(
                1 + self.variable_distr_A2[var].pdf(row, var, size))
        return row

    def evaluate(self, X_test):
        X_test['output_A1'] = 1.0
        X_test['output_A2'] = 1.0
        X_test.apply(self.calculate_bayes, axis=1)
        output = X_test['output_A1'] < X_test['output_A2']
        return [1 if i else 0 for i in list(output)]


class GaussianBayesClassifier:
    def __init__(self):
        self.prior_probabilities = {}
        self.variable_distr_A1 = {}
        self.variable_distr_A2 = {}
        self.variables = []

    def fit(self, X_train, y_train):
        y_col = y_train.columns[-1]
        self.variables = X_train.columns
        self.prior_probabilities = dict.fromkeys(y_train[y_col].unique())
        for cls in self.prior_probabilities.keys():
            count = len(y_train[y_train[y_col] == cls])
            self.prior_probabilities[cls] = count / len(y_train)
        self.variable_distr_A1 = dict.fromkeys(X_train.columns)
        self.variable_distr_A2 = dict.fromkeys(X_train.columns)
        train_A1 = X_train[y_train[y_col] == 0]
        train_A2 = X_train[y_train[y_col] == 1]
        for var in X_train.columns:
            mu_A1 = train_A1[var].mean()
            sigma_A1 = train_A1[var].std()
            self.variable_distr_A1[var] = statistics.NormalDist(mu=mu_A1,
                                                                sigma=sigma_A1)
            mu_A2 = train_A2[var].mean()
            sigma_A2 = train_A2[var].std()
            self.variable_distr_A2[var] = statistics.NormalDist(mu=mu_A2,
                                                                sigma=sigma_A2)

    def calculate_bayes(self, row):
        for var in self.variables:
            row['output_A1'] += log(
                1 + self.variable_distr_A1[var].pdf(row[var]))
            row['output_A2'] += log(
                1 + self.variable_distr_A2[var].pdf(row[var]))
        return row

    def evaluate(self, X_test):
        X_test['output_A1'] = 1.0
        X_test['output_A2'] = 1.0
        X_test.apply(self.calculate_bayes, axis=1)
        output = X_test['output_A1'] < X_test['output_A2']
        return [1 if i else 0 for i in list(output)]


def scatter_plot(X, y, label):
    A1_X = X[y['y'] == 0]
    A2_X = X[y['y'] == 1]
    plt.scatter(x=A1_X['x1'], y=A1_X['x2'], c='red')
    plt.scatter(x=A2_X['x1'], y=A2_X['x2'], c='blue')
    plt.title(label)
    plt.show()


if __name__ == '__main__':
    print("\n2d_dataset:")

    test_size = 0.5
    items_A1 = 100
    items_A2 = 100

    synthetic_A1 = np.random.multivariate_normal(mean=[0, 0],
                                                 cov=[[2, -1], [-1, 2]],
                                                 size=items_A1)
    synthetic_A2 = np.random.multivariate_normal(mean=[2, 2],
                                                 cov=[[1, 0], [0, 1]],
                                                 size=items_A2)

    # plt.scatter(x=list(zip(*synthetic_A1))[0],
    #             y=list(zip(*synthetic_A1))[1], c='red', label='A1')
    # plt.scatter(x=list(zip(*synthetic_A2))[0],
    #             y=list(zip(*synthetic_A2))[1], c='blue', label='A2')
    # plt.legend()
    # plt.show()

    synthetic = [(*s, 0) for s in synthetic_A1] + \
                [(*s, 1) for s in synthetic_A2]

    synthetic = pd.DataFrame(synthetic, columns=['x1', 'x2', 'y'])

    X_train, X_test, y_train, y_test = train_test_split(
        synthetic[['x1', 'x2']],
        synthetic[['y']],
        test_size=test_size,
        stratify=synthetic[['y']])

    # scatter_plot(X_train, y_train, 'Train set')
    # scatter_plot(X_test, y_test, 'Test set')

    print('GaussianBayes:')
    bc = GaussianBayesClassifier()
    bc.fit(X_train, y_train)
    pred = bc.evaluate(X_test)
    real = list(y_test['y'])
    confusion = Confusion.from_wrong_preds(['A1', 'A2'], pred, real,
                                           {'A1': 50, 'A2': 50})
    print(confusion)

    print('ParzenBayes:')
    pc = ParzenBayesClassifier()
    pc.fit(X_train, y_train)
    pred = pc.evaluate(X_test)
    real = list(y_test['y'])
    confusion = Confusion.from_wrong_preds(['A1', 'A2'], pred, real,
                                           {'A1': 50, 'A2': 50})
    print(confusion)

    #####################################################################################

    print("\n5d_non-gaussian_dataset:")
    # uniform points around (.5, .5, .5, .5, .5)
    synthetic_5d_A = np.random.uniform(low=0.0, high=1.0, size=(100, 5))
    # uniform points around (1., 1., 1., 1., 1.)
    synthetic_5d_B = np.random.uniform(low=0.5, high=1.5, size=(100, 5))

    synthetic_5d = [(*s, 0) for s in synthetic_5d_A] + \
                   [(*s, 1) for s in synthetic_5d_B]

    synthetic_5d = pd.DataFrame(synthetic_5d, columns=['x1', 'x2', 'x3', 'x4', 'x5', 'y'])
    X_train, X_test, y_train, y_test = train_test_split(
        synthetic_5d[['x1', 'x2', 'x3', 'x4', 'x5']],
        synthetic_5d[['y']],
        test_size=test_size,
        stratify=synthetic_5d[['y']])

    print('GaussianBayes:')
    bc = GaussianBayesClassifier()
    bc.fit(X_train, y_train)
    pred = bc.evaluate(X_test)
    real = list(y_test['y'])
    confusion = Confusion.from_wrong_preds(['A1', 'A2'], pred, real,
                                           {'A1': 50, 'A2': 50})
    print(confusion)

    print('ParzenBayes:')
    pc = ParzenBayesClassifier()
    pc.fit(X_train, y_train)
    pred = pc.evaluate(X_test)
    real = list(y_test['y'])
    confusion = Confusion.from_wrong_preds(['A1', 'A2'], pred, real,
                                           {'A1': 50, 'A2': 50})
    print(confusion)

    #####################################################################################

    print("\ncancer_dataset:")
    test = sio.loadmat(r'C:\Users\piotr\Desktop\dane1-10\dane10.mat')
    test_y = pd.DataFrame(test['testowy'][0][0][0].transpose())
    test_X = pd.DataFrame(test['testowy'][0][0][1].transpose())

    train_y = pd.DataFrame(test['uczacy'][0][0][0].transpose())
    train_X = pd.DataFrame(test['uczacy'][0][0][1].transpose())

    print('GaussianBayes:')
    bc = GaussianBayesClassifier()
    bc.fit(train_X, train_y)
    pred = bc.evaluate(test_X)
    real = list(test_y[0])
    confusion = Confusion.from_wrong_preds(['0', '1'], pred, real,
                                           {'0': 20, '1': 20})
    print(confusion)

    print('ParzenBayes:')
    pc = ParzenBayesClassifier()
    pc.fit(train_X, train_y)
    pred = pc.evaluate(test_X)
    real = list(test_y[0])
    confusion = Confusion.from_wrong_preds(['0', '1'], pred, real,
                                           {'0': 20, '1': 20})
    print(confusion)
