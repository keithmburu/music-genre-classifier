"""
Categorical Naive Bayes from scratch and using sklearn's implementation
Adapted from lab 5
Author: Keith Mburu
Author: Matt Gusdorff
Date: 12/17/2021
"""

import os
import math
from collections import OrderedDict
from numpy.lib.type_check import real
import util
import numpy as np
from entropy import Entropy
import matplotlib.pyplot as plt 
import seaborn as sns
import sklearn.naive_bayes as nb
import pandas as pd
import ConfusionMatrix as conf

class NaiveBayes:

    def __init__(self, train_partition, train_file, percentage_of_features=100):
        """
        Initializes model that is used to classify examples
        train_partition: object containing training data for model creation
        best_features_only: bool indicating whether we only want best features
        """
        # dictionary of features and possible feature values
        F = train_partition.F
        # number of classes
        K = train_partition.K
        
        # features being considered
        self.feature_names = relevant_features(train_file, F.keys(), percentage_of_features)

        # uncomment for unsorted features

        # features = list(F.keys())
        # random.shuffle(features)
        # self.feature_names = features[:math.ceil((percentage_of_features/100)*len(F.keys()))]

        if os.path.exists(f"objects/nb_{str(percentage_of_features)}%_{train_file}.pkl"):
            print("\nloading prior,likelihood\n")
            self.prior, self.likelihood = util.object_load(f"nb_{str(percentage_of_features)}%_" + train_file)     
        elif os.path.exists(f"objects/nb_100%_{train_file}.pkl"):
            print("\nloading prior,likelihood\n")
            self.prior, self.likelihood = util.object_load(f"nb_100%_{train_file}")
        else:
            # list containing prior p(y=k) for each class k
            self.prior = [0] * K
            # list of k dictionaries corresponding to k classes, each with j
            # dictionaries corresponding to j features, and containing the
            # likelihood of each feature value p(x_v | y=k)
            self.likelihood = [{}] * K
        
            for k in range(K):
                # number of examples of class k
                N_k = 0
                print("\nconstructing prior\n")
                for example in train_partition.data:
                    if example.label == k:
                        N_k += 1
                # updating prior p(y=k)
                self.prior[k] = math.log(N_k + 1) - math.log(train_partition.n + K)

                # dictionary corresponding to y=k
                y_k_dict = OrderedDict()
                print("\nconstructing likelihood\n")
                for j in range(len(self.feature_names)):
                    # dictionary corresponding to feature j
                    feature_value_dict = OrderedDict()
                    # for each possible feature value
                    for v in range(len(F[self.feature_names[j]])):
                        # number of examples of class k that have feature j=v
                        N_kjv = 0
                        for example in train_partition.data:
                            if example.label == k and \
                            example.features[self.feature_names[j]] == \
                            F[self.feature_names[j]][v]:
                                N_kjv += 1
                        # updating likelihood p(x_v | y=k)
                        feature_value_dict[F[self.feature_names[j]][v]] = \
                            math.log(N_kjv + 1) - math.log(N_k + \
                                len(F[self.feature_names[j]]))
                    y_k_dict[self.feature_names[j]] = feature_value_dict
                self.likelihood[k] = y_k_dict
            print("\nstoring prior, likelihood\n")
            util.object_store([self.prior, self.likelihood], f"nb_{str(percentage_of_features)}%_{train_file}")

    def classify(self, x_test):
        """
        Predicts the class of an example based on its feature real_values
        x_test: example's dictionary of features
        """
        # predicted label (default = -1)
        k_hat = -1
        # initializing arbitrary small log posterior probability
        y_hat = -1000000
        for k in range(len(self.prior)):
            y_k_hat = self.prior[k]
            for feature in self.feature_names:
                # log addition
                y_k_hat += self.likelihood[k][feature][x_test[feature]]
            # finding class with max probability
            if y_k_hat > y_hat:
                k_hat = k
                y_hat = y_k_hat
        return k_hat


def relevant_features(train_file, all_features, percentage_of_features):
    """
    Returns list of features to be used based on percentage to be tested
    """
    if percentage_of_features != 100:
        print("\nloading sorted features\n")
        tokens = train_file.split("_")
        n = tokens[3][1:]
        try:
            sorted_features = util.object_load(f"100%_disc_features_sorted_n{n}")
        except:
            print(f"creating sorted features list for n={n}")
            try:
                full_partition = util.object_load(f"1_100%_features_n{n}_3_sec_shuffled.csv")
            except:
                print(f"read in csv file with 100% of the features for n={n} first")
                exit()
            sorted_features = Entropy(full_partition).sort_features()
            print("\nstoring sorted features\n")
            util.object_store(sorted_features, f"100%_disc_features_sorted_n{n}")
        return sorted_features[:math.ceil((percentage_of_features/100)*len(sorted_features))]
    else:
        return list(all_features)

def scratch_test(n, train_partition, train_file, test_partition_refeaturized, start, stop, step):
    """
    Testing from scratch implementation at different percentages
    """
    x_vals = []
    y_vals = []
    for percentage in range(start, stop, step):
        print(f"\nPercentage: {percentage}\n")
        # initializing model
        nb_model = NaiveBayes(train_partition, train_file, percentage)
        print("\nmodel initialized\n")
        real_values = []
        model_values = []
        print("classifying")
        for example in test_partition_refeaturized.data:
            real_values.append(example.label)
            # classifying examples
            model_values.append(nb_model.classify(example.features))

        # presenting accuracy and confusion matrix
        cm = conf.ConfusionMatrix(util.GENRES, real_values, model_values)
        print(f"\nAccuracy: {str(cm.accuracy())}({str(cm.accuracy() * test_partition_refeaturized.n)} out of {str(test_partition_refeaturized.n)} correct)")
        print(cm)
        cm_df = pd.DataFrame(cm.matrix, index = util.GENRES, columns = util.GENRES)
        plt.figure(figsize = (7,6))
        sns.heatmap(cm_df, annot=True, cmap="Greens", fmt='g')
        plt.savefig(f"figures/nb_scratch_n{n}_{percentage}%_cm.png")
        plt.show()

        x_vals.append(percentage)
        y_vals.append(cm.accuracy()*100)
    return x_vals, y_vals

def sklearn_test(n, disc, train_partition, train_file, test_partition_refeaturized, test_file, start, stop, step):
    """
    Testing sklearn implementation at different percentages
    """
    x_vals = []
    y_vals = []
    for percentage in range(start, stop, step):
        print(f"\nPercentage: {percentage}%\n")
        # features being considered
        feature_names = relevant_features(train_file, train_partition.F.keys(), percentage)
        # initializing model
        clf = nb.CategoricalNB()
        print("\nmodel initialized\n")
        if os.path.exists(f"objects/X_y_{disc}_{percentage}%_{train_file}.pkl"):
            print("\nloading X, y\n")
            X, y = util.object_load(f"X_y_{disc}_{percentage}%_{train_file}")
        else:  
            # Converting partitions to numpy array format
            X = np.zeros([train_partition.n, len(feature_names)])
            y = np.zeros([train_partition.n, 1])
            for i, example in enumerate(train_partition.data):
                print(f"\ninitializing train X and y, {i}\n")
                j = 0
                for feature in example.features:
                    if feature in feature_names:
                        if disc:
                            if example.features[feature] == "True":  
                                X[i, j] = 1
                        else:
                            X[i, j] = example.features[feature]
                        j += 1
                y[i] = example.label
            util.object_store([X, y], f"X_y_{disc}_{percentage}%_{train_file}")

        print("\nfitting model\n")
        clf.fit(X, y)
        
        real_values = []
        if os.path.exists(f"objects/X0_y0_0_{percentage}%_{test_file}.pkl"):
            print("\nloading X0, y0\n")
            X_0, y_0 = util.object_load(f"X0_y0_0_{percentage}%_{test_file}")
        else:  
            X_0 = np.zeros([test_partition_refeaturized.n, len(feature_names)])
            y_0 = np.zeros([test_partition_refeaturized.n, 1])
            # classifying examples
            for i, example in enumerate(test_partition_refeaturized.data):
                print(f"\ninitializing test X0 and y0, {i}\n")
                j = 0
                for feature in example.features:
                    if feature in feature_names:
                        if disc:
                            if example.features[feature] == "True":  
                                X_0[i, j] = 1
                        else:
                            X_0[i, j] = example.features[feature]
                        j += 1
                y_0[i] = example.label
                real_values.append(example.label)
            util.object_store([X_0, y_0], f"X0_y0_0_{percentage}%_{test_file}")
        
        model_values = clf.predict(X_0).astype(int)

        # presenting accuracy and confusion matrix
        cm = conf.ConfusionMatrix(util.GENRES, real_values, model_values)
        print(f"\nAccuracy: {str(cm.accuracy())}({str(cm.accuracy() * test_partition_refeaturized.n)} out of {str(test_partition_refeaturized.n)} correct)")
        print(cm)
        cm_df = pd.DataFrame(cm.matrix, index = util.GENRES, columns = util.GENRES)
        plt.figure(figsize = (7,6))
        sns.heatmap(cm_df, annot=True, cmap="Greens")
        plt.savefig(f"figures/nb_sklearn_n{n}_{percentage}_cm.png")
        plt.show()

        x_vals.append(percentage)
        y_vals.append(cm.accuracy()*100)
    return x_vals, y_vals

def plot(x_vals, y_vals, method, n):
    """
    Plotting accuracy against percentage of features used
    """
    plt.plot(x_vals, y_vals)
    plt.xlim([10, 100])
    plt.ylim([0, 100])
    plt.title("Naive Bayes accuracy vs percentage of features used")
    plt.xlabel("Percentage of features used")
    plt.ylabel("Classification accuracy")
    plt.savefig(f"figures/nb_{method}_n{n}_accuracy.png")
    plt.show()
    plt.clf()
