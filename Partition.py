"""
Example and partition classes with labels
Adapted from Lab 5 (Sara Mathieson)
Author: Keith Mburu
Author: Matt Gusdorff
Date: 12/17/2021
"""

class Example:

    def __init__(self, features, label):
        # dictionary. key=feature name: value=feature value for this example
        self.features = features
        self.label = label

class Partition:

    def __init__(self, data, F, K):
        # list of Examples
        self.data = data
        self.n = len(self.data)

        # dictionary. key=feature name: value=list of possible values
        self.F = F

        # number of classes
        self.K = K

        temp = []
        for example in self.data:
            if example.label not in temp:
                temp.append(example.label)

        self.labels = temp
