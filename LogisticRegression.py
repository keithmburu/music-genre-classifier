import math
import numpy as np

import entropy

class LogisticRegression:

    def __init__(self, partition):
        self.partition = partition
        self.X, self.y = self.get_matrices()

    def model(self,  label):
        '''
        Builds a logistic regression model for a specific label (genre, blues yes or no?)
        '''
        labels = self.genrify(label)
        w = self.SGD(self.X, labels)

        def model(x):
            dot = np.dot(w, x)
            try:
                return 1/(1+math.exp(-dot))
            except:
                if -dot > 0:
                    return 0
                else:
                    return 1

        return model

    def genrify(self, label):
        labels = [0 for label in self.y]
        for i in range(len(self.y)):
            if self.y[i] == label:
                labels[i] = 1

        return labels

    def get_matrices(self):
        X = np.zeros([self.partition.n, len(self.partition.F)])
        y = np.zeros([self.partition.n])
        for i in range(self.partition.n):
            y[i] = self.partition.data[i].label
            j=0
            for feature in self.partition.F:
                X[i][j] = self.partition.data[i].features[feature]
                j += 1

        self.normalize(X)
        X = self.add_ones(X)

        return X, y

    def normalize(self, matrix):
        '''
        Normalizes the data in matrix
        matrix: the numpy matrix containing the data to be normalized
        '''
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        try:
            for i in range(matrix.shape[1]):
                matrix[:, i] -= mean[i]
                matrix[:, i] *= (1 / std[i])
        except:
            matrix -= mean
            matrix *= (1 / std)

    def add_ones(self, matrix):
        '''
        Concatenates a column of ones to the start of matrix
        matrix: the numpy matrix to add ones to
        returns the numpy matrix with the column of ones
        '''
        ones = np.ones((len(matrix), 1))
        new = np.concatenate((ones, matrix), 1)
        return new




    def model_eq(self, w, x):
        dot = np.dot(w,x)
        try:
            return 1/(1+math.exp(-dot))
        except:
            if -dot > 0:
                return 0
            else:
                return 1

    def cost(self, labels, w, X):
        sum = 0
        for i in range(len(labels)):
            if labels[i] == 1:
                sum += math.log(self.model_eq(w, X[i]))
            else:
                sum += math.log(1-self.model_eq(w, X[i]))
        return -sum

    def SGD(self, X, labels, eps=1e-10, t_max=10000):
        alpha = 0.1
        w = np.zeros(X.shape[1])
        w = w - alpha * self.model_eq(w, X[0] - labels[0]) * X[0]
        old_cost = self.cost(labels, w, X)
        new_cost = 0
        t = 1
        i = 1
        while t <= t_max:
            w = w - alpha * self.model_eq(w, X[i] - labels[i]) * X[i]
            new_cost = self.cost(labels, w, X)
            if abs(new_cost-old_cost) <= eps:
                break

            old_cost = new_cost

            if t % 100 == 0:
                alpha /= 10

            t += 1
            i += 1
            if i >= len(labels):
                i = 0

        return w
