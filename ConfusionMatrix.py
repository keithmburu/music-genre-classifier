"""
Confusion Matrix class
Author: Keith Mburu
Author: Matt Gusdorff
Date: 12/17/2021
"""

import numpy as np

class ConfusionMatrix:

    def __init__(self, labels, true, guessed):
        self.labels = labels
        self.true = true
        self.guessed = guessed
        self.matrix = self.generate_matrix()

    def generate_matrix(self):
        '''
        generates the confusion matrix
        returns the confusion matrix based on the true and guessed labels
        '''
        label_dict = {}
        for i in range(len(self.labels)):
            label_dict[self.labels[i]] = i

        mat = np.zeros([len(self.labels), len(self.labels)])

        for i in range(len(self.guessed)):
            try:
                mat[label_dict[self.true[i]], label_dict[self.guessed[i]]] += 1
            except:
                mat[self.true[i], self.guessed[i]] += 1

        return mat.astype(int)

    def accuracy(self):
        '''
        Calculates the accuracy (diagonal entries/total)
        returns the accuracy
        '''
        correct = 0
        for i in range(len(self.matrix)):
            correct += self.matrix[i][i]

        return correct/len(self.guessed)


    def __str__(self):
        '''
        tostring function for the matrix
        returns a properly formatted string representing the matrix
        '''
        biggest = max(self.labels, key=lambda x: len(x))

        #basically all of the " " indicate spacing depending on length of names/numbers
        string = " " * len(biggest) + " "
        for label in self.labels:
            string += label + " "
        string += '\n'
        for i in range(len(self.matrix)):
            string += self.labels[i] + " " * (len(biggest) - len(self.labels[i])) + "| "
            for j in range(len(self.matrix[i])):
                string += str(self.matrix[i][j]) + " " * ( len(self.labels[j]) - len(str(self.matrix[i][j])) + 1)
            string += '\n'

        return string

