"""
Entropy class for sorting features based on information gain
Author: Keith Mburu
Author: Matt Gusdorff
Date: 12/17/2021
"""

import math

class Entropy:
    def __init__(self, partition):
        self.partition = partition
        self.entropy = self.calc_entropy()
        self.prob_fv_dict = self.gen_prob_fv_dict()
        self.prob_k_fv_dict = self.gen_prob_k_fv_dict()
        self.entropy_fv_dict = self.gen_entropy_fv_dict()
        self.conditional_entropy_dict = self.gen_conditional_entropy_dict()
        self.sorted_features = self.sort_features()

    def sort_features(self):
        '''
        sorts features based on their information gain
        returns the sorted list
        '''
        features = []
        for feature in self.partition.F:
            features.append(feature)

        features.sort(key=lambda x: self.info_gain(x), reverse=True)
        return features


    def info_gain(self, feature):
        '''
        Info gain = entropy - conditional entropy of feature
        feature: the feature that we calculate the conditional entropy for
        return the info gain for feature
        '''
        return self.entropy - self.conditional_entropy_dict[feature]

    def calc_entropy(self):
        '''
        returns the entropy of the data
        '''
        probs = {}
        for example in self.partition.data:
            if example.label in probs:
                probs[example.label] += 1
            else:
                probs[example.label] = 1

        for key in probs:
            probs[key] /= self.partition.n

        entropy = 0

        for key in probs:
            #equation for entropy
            entropy += probs[key]*-math.log2(probs[key])

        return entropy


    def gen_conditional_entropy_dict(self):
        '''
        generates a dictionary of feature keys and conditional entropy values
        returns the dictionary
        '''
        dict = {}
        for feature in self.partition.F:
            dict[feature] = self.calc_conditional_entropy(feature)

        return dict

    def calc_conditional_entropy(self, feature):
        '''
        calculates the conditional entropy for a feature
        feature: the feature for which we calculate the conditional entropy
        return the conditional entropy
        '''
        entropy = 0
        for value in self.partition.F[feature]:
            entropy += self.prob_fv(feature, value) * self.entropy_fv_dict[(feature, value)]

        return entropy

    def gen_entropy_fv_dict(self):
        '''
        generates a dictionary of feature,value tuples to entropy
        return the dictionary
        '''
        dict = {}
        for feature in self.partition.F:
            for value in self.partition.F[feature]:
                dict[(feature, value)] = self.calc_entropy_fv(feature, value)

        return dict

    def calc_entropy_fv(self, feature, value):
        '''
        calculates the entropy for a feature,value tuple
        feature, value: the feature and value in said tuple
        return the entrop
        '''
        entropy = 0
        for label in self.partition.labels:
            if self.prob_k_fv(label,feature,value) != 0:
                entropy += self.prob_k_fv(label,feature,value)*(-math.log2(self.prob_k_fv(label,feature,value)))

        return entropy

    def gen_prob_fv_dict(self):
        '''
        generates a dictionary of feature, value tuples to probabilities of f=v
        return the dictionary
        '''
        dict = {}
        for feature in self.partition.F:
            for value in self.partition.F[feature]:
                dict[(feature, value)] = self.prob_fv(feature, value)

        return dict

    def prob_fv(self, feature, value):
        '''
        Calculates the probability that feature has value value
        feature: the feature
        value: the specific value
        return the probability
        '''
        total = 0
        for example in self.partition.data:
            if example.features[feature] == value:
                total += 1

        return total/self.partition.n

    def gen_prob_k_fv_dict(self):
        '''
        generates a dictionary of (label, feature, value) tuples to probabilities
        that f=v and label=k
        '''
        dict = {}
        for label in self.partition.labels:
            for feature in self.partition.F:
                for value in self.partition.F[feature]:
                    dict[(label, feature, value)] = self.prob_k_fv(label, feature, value)

    def prob_k_fv(self, label, feature, value):
        '''
        Calculate the probability that an example has label label with feature=value
        label,feature,value: the tuple we base the probability off of
        return the probability
        '''
        total = 0
        for example in self.partition.data:
            if example.features[feature] == value and example.label == label:
                total += 1

        return total/self.partition.n
