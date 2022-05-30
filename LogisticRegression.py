"""
Logistic Regression class for test classification
Author: Keith Mburu
Author: Matt Gusdorff
Date: 12/17/2021
"""

import math
import numpy as np
import util

class LogisticRegression:

    def __init__(self, partition, allowed_features=None):
        self.partition = partition
        self.allowed_features = allowed_features

        if self.allowed_features is None:
            self.allowed_features = list(partition.F.keys())

        self.X, self.y = self.get_matrices()

    def model(self, label):
        '''
        Builds a logistic regression model for a specific label (genre, blues yes or no?)
        label: the genre we are basing the model on
        returns a function represeting a logistic regression model
        '''
        labels = self.genrify(label)
        #get the weight vector from SGD
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
        '''
        Alters the data in y to be binary (is it the label, 0 or 1)
        label: the genre that we are "genrifying" y off of
        return the new y
        '''
        labels = [0 for label in self.y]
        for i in range(len(self.y)):
            if self.y[i] == util.GENRE_LABELS[label]:
                labels[i] = 1

        return labels

    def get_matrices(self):
        '''
        Converts a partition to a X matrix of features and a y vector of labels
        return a tuple of X,y
        '''
        X = np.zeros([self.partition.n, len(self.allowed_features)])
        y = np.zeros([self.partition.n])
        for i in range(self.partition.n):
            y[i] = self.partition.data[i].label
            j = 0
            for feature in self.allowed_features:
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
        '''
        A function for the logistic regression model
        w: the vector of weights
        x: the vector of numerical features
        '''
        dot = np.dot(w,x)
        try:
            return 1/(1+math.exp(-dot))
        except: #if the dot product is big, there'll be an error
            if -dot > 0: #if the dot is super big, then denominator -> infinity
                return 0
            else:
                return 1

    def cost(self, labels, w, X):
        '''
        Calculates the cost of the logistic regression model
        labels: the true labels for the data
        w: the weight vector
        X: the feature matrix
        returns the cost of applying a model based on w to the dataset X
        '''
        sum = 0
        for i in range(len(labels)):
            if labels[i] == 1:
                try:
                    sum += math.log(self.model_eq(w, X[i]))
                except: #more error handling if the dot product is super large
                    sum += math.log(self.model_eq(w, X[i]) - 0.0000000000000000001)
            else:
                try:
                    sum += math.log(1-self.model_eq(w, X[i]))
                except:
                    sum += math.log(1 - self.model_eq(w, X[i]) + 0.0000000000000000001)
        return -sum

    def SGD(self, X, labels, eps=1e-10, t_max=10000):
        '''
        Application of the SGD algorithm for finding a weight vector for the logistic regression model
        X: the matrix of examples
        labels: the true labels of the examples
        eps: the maximum difference between iterations to stop the algorithm
        t_max: the maximum amount of iterations
        returns the weight vector
        '''
        alpha = 0.1
        w = np.zeros(X.shape[1])
        #application of the SGD equation
        w = w - alpha * (self.model_eq(w, X[0]) - labels[0]) * X[0]
        old_cost = self.cost(labels, w, X)
        new_cost = 0
        t = 1
        i = 1

        #there was some odd bouncing, so we keep track of the minimum cost
        min_cost = old_cost
        min_w = w
        while t <= t_max:
            #iterate w
            w = w - alpha * (self.model_eq(w, X[i]) - labels[i]) * X[i]
            new_cost = self.cost(labels, w, X)

            #reassign min_cost
            if new_cost < min_cost:
                min_cost = new_cost
                min_w = w

            #check change in cost
            if abs(new_cost-old_cost) <= eps:
                break

            old_cost = new_cost

            alpha = 1/(t ** 0.25) #change alpha

            t += 1
            i += 1
            if i >= len(labels):
                i = 0

        #print("FINAL:", min_cost, min_w)
        return min_w
