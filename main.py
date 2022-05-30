"""
A driver file for implementing our various methods
Author: Keith Mburu
Author: Matt Gusdorff
Date: 12/17/2021
"""

import ConfusionMatrix as conf
import Entropy as ent
import LogisticRegression as lr

import os
import math
import cont_to_disc
import matplotlib.pyplot as plt
import numpy as np
import util
import Naive_bayes as nb
import dim_reduction as dr

def main():
    if not os.path.exists("objects"):
        os.mkdir("objects")
    
    # PCA, t-SNE

    cont_to_disc = 1
    n = 1600
    filename, partition = util.read_csv(f"data/features_n{n}_3_sec_shuffled.csv", cont_to_disc)

    pca_x, y = dr.setup("pca", filename, partition, cont_to_disc)
    tsne_x, y = dr.setup("tsne", filename, partition, cont_to_disc)
    dr.plot("pca", pca_x, cont_to_disc, y, n)
    dr.plot("tsne", tsne_x, cont_to_disc, y, n)
    
    # Naive Bayes

    n = 80
    disc = 1
    train_filename = f"data/features_n{n}_3_sec_shuffled.csv"
    test_filename = f"data/features_n{int(n/4)}_3_sec_shuffled.csv"
    train_file, train_partition = util.read_csv(train_filename, disc, 100)
    test_file, test_partition = util.read_csv(test_filename, 0, 100)
    if disc:
        test_partition_refeaturized = cont_to_disc.refeaturize(train_partition, test_partition, test_file)
    else:
        test_partition_refeaturized = test_partition
        
    # from scratch implementation
    x_vals, y_vals = nb.scratch_test(n, train_partition, train_file, test_partition_refeaturized, 10, 101, 10)
    nb.plot(x_vals, y_vals, "scratch", n)
    #sklearn implementation
    x_vals, y_vals = nb.sklearn_test(n, disc, train_partition, train_file, test_partition_refeaturized, test_file, 10, 101, 10)
    nb.plot(x_vals, y_vals, "sklearn", n)

    # Logistic Regression

    print("\n===== BEGINNING DATA COMPILATION FOR LOGISTIC REGRESSION =====\n")
    print("Reading continuous training data")
    ctrain_partition = util.read_csv("data/features_n800_3_sec_shuffled.csv", False)[1]
    print("Reading continuous testing data")
    ctest_partition = util.read_csv("data/features_n200_3_sec_shuffled.csv", False)[1]
    print("Building logistic regression model generator")
    #graph_features_TP(ctrain_partition, ctest_partition)
    #graph_features_committee_accuracy(ctrain_partition, ctest_partition)
    #graph_features_accuracy(ctrain_partition, ctest_partition)
    #graph_features_average_accuracy(ctrain_partition, ctest_partition)


def graph_features_accuracy(train_partition, test_partition):
    '''
    Saves graphs of number of features vs accuracy for each genre
    train_partition: the partition to train the logistic regression models on
    test_partition: partition containing testing data
    '''
    N = [i for i in range(1, len(util.SORTED_CONTINUOUS_FEATURES) + 1)]
    genre_accs = {}
    for genre in util.GENRES:
        genre_accs[genre] = []

    for i in range(1, len(util.SORTED_CONTINUOUS_FEATURES) + 1):
        #specifying the first i best features to use
        allowed_features = util.SORTED_CONTINUOUS_FEATURES[0:i]
        #making our model
        lr_generator = lr.LogisticRegression(train_partition, allowed_features)
        testX, testY = get_matrices(test_partition, allowed_features)
        models = {}
        for genre in util.GENRES:
            models[genre] = lr_generator.model(genre)
            #producing a confusion matrix from which we get the accuracy
            mat = genre_matrix(testX, testY, genre, models[genre])
            genre_accs[genre].append(mat.accuracy())

        print("complete N =", i)

    for genre in genre_accs:
        plt.xlabel("Number of best features")
        plt.ylabel("Classification Accuracy")
        plt.title(f"Accuracy for {genre} vs. Features used")
        plt.plot(N, genre_accs[genre])
        plt.savefig(f"figures/log_reg_{genre}_acc.pdf", format='pdf')
        plt.clf()

def graph_features_average_accuracy(train_partition, test_partition):
    '''
    Saves a graph of number of features vs average accuracy across all genre models
    train_partition: the partition to train the logistic regression models on
    test_partition: partition containing testing data
    '''
    N = [i for i in range(1, len(util.SORTED_CONTINUOUS_FEATURES) + 1)]
    accs = [0 for i in range(len(N))]

    for i in range(1, len(util.SORTED_CONTINUOUS_FEATURES) + 1):
        allowed_features = util.SORTED_CONTINUOUS_FEATURES[0:i]
        #making our model based on the first i best features
        lr_generator = lr.LogisticRegression(train_partition, allowed_features)
        testX, testY = get_matrices(test_partition, allowed_features)
        models = {}

        for genre in util.GENRES:
            models[genre] = lr_generator.model(genre)
            mat = genre_matrix(testX, testY, genre, models[genre])
            #recall that i is never 0, so we subtract one in our list
            accs[i - 1] += mat.accuracy()

        accs[i - 1] /= len(util.GENRES)
        print("complete N =", i)

    plt.xlabel("Number of best features")
    plt.ylabel("Classification Accuracy")
    plt.title(f"Average model accuracy vs. Features used")
    plt.plot(N, accs)
    plt.savefig(f"figures/log_reg_avg_acc.pdf", format='pdf')
    plt.clf()

def graph_features_TP(train_partition, test_partition):
    '''
    Saves graphs of number of features vs "true-positive" rate for each genre
    train_partition: the partition to train the logistic regression models on
    test_partition: partition containing testing data
    '''
    N = [i for i in range(1, len(util.SORTED_CONTINUOUS_FEATURES) + 1)]
    genre_TPs = {}
    for genre in util.GENRES:
        genre_TPs[genre] = []

    for i in range(1, len(util.SORTED_CONTINUOUS_FEATURES) + 1):
        allowed_features = util.SORTED_CONTINUOUS_FEATURES[0:i]
        #same deal as the other methods
        lr_generator = lr.LogisticRegression(train_partition, allowed_features)
        testX, testY = get_matrices(test_partition, allowed_features)
        models = {}
        for genre in util.GENRES:
            models[genre] = lr_generator.model(genre)
            mat = genre_matrix(testX, testY, genre, models[genre])
            genre_TPs[genre].append(mat.matrix[1][1] / sum(mat.matrix[1]) if sum(mat.matrix[1]) != 0 else 0)


        print("complete N =",i)

    for genre in genre_TPs:
        plt.xlabel("Number of best features")
        plt.ylabel("True Positive rate")
        plt.title(f"True positive rate for {genre} vs. Features used")
        plt.plot(N, genre_TPs[genre])
        plt.savefig(f"figures/log_reg_{genre}.pdf", format='pdf')
        plt.clf()

def graph_features_committee_accuracy(train_partition, test_partition):
    '''
    Saves a graph of number of features vs accuracy of the committee method
    train_partition: the partition to train the logistic regression models on
    test_partition: partition containing testing data
    '''
    N = [i for i in range(1, len(util.SORTED_CONTINUOUS_FEATURES) + 1)]
    accuracies = []
    for i in range(1, len(util.SORTED_CONTINUOUS_FEATURES) + 1):
        allowed_features = util.SORTED_CONTINUOUS_FEATURES[0:i]
        lr_generator = lr.LogisticRegression(train_partition, allowed_features)
        testX, testY = get_matrices(test_partition, allowed_features)
        models = {}
        for genre in util.GENRES:
            models[genre] = lr_generator.model(genre)

        #note that we are making a matrix from the committee method this time
        mat = committee_matrix(testX, testY, models)
        accuracies.append(mat.accuracy())


        print("complete N =", i)

    plt.plot(N, accuracies)

    plt.xlabel("Number of best features")
    plt.ylabel("Accuracy")
    plt.title("Committee Approach")
    plt.savefig("figures/log_reg_committee.pdf", format='pdf')
    plt.show()

def simplify_features(features):
    '''
    Takes a set of the best discrete features and converts them to the best continuous features
    features: the list of the best discrete features
    returns the sorted list of best continuous features
    '''
    best = []
    weights = {}
    for feature in features:
        #here, we need to take of the substrings of the features i.e before the <=
        i = feature.index("<=")
        name = feature[0:i]
        if name not in weights:
            weights[name] = 0

    for i in range(len(features)):
        index = features[i].index("<=")
        name = features[i][0:index]
        #here we go through the list and if we see a feature (before the <=) pop up, then we
        #increase its weight according to how far in the list it is (far = worse)
        weights[name] += i

    for name in weights:
        best.append((name, weights[name]))

    #sort by weight
    best.sort(key=lambda x: x[1])

    return [x[0] for x in best]

def committee_matrix(testX, testY, models):
    '''
    Creates a confusion matrix based on the committee method
    testX: a matrix of the test example features
    testY: vector of the test example labels
    models: a dictionary of models for each genre
    returns a confusion matrix
    '''
    guessed = []
    true = []

    for x in testX:
        guessed.append(committee(x, models))

    for y in testY:
        true.append(util.GENRE_LABELS_R[y])

    return conf.ConfusionMatrix(util.GENRES, true, guessed)

def committee(x, models):
    '''
    Returns the maximum probability from the models on example x
    x: the vector of features
    models: the dictionary of models for each genre
    returns the genre label of the maximum model probability
    '''
    max = -1
    maxGenre = ""
    for model in models:
        if models[model](x) > max:
            max = models[model](x)
            maxGenre = model

    return maxGenre

def genre_matrix(testX, testY, genre, model):
    '''
    Creates a confusion matrix for a model of a specific genre
    testX: the matrix of example features
    testY: the vector of example labels
    genre: the genre corresponding to the model
    model: a logistic regression model for genre
    returns a confusion matrix for genre's model
    '''
    true = []
    guessed = []

    for i in range(len(testX)):
        if round(model(testX[i])) == 1:
            guessed.append(genre)
        else:
            guessed.append(f"not {genre}") #string formatting

    for y in testY:
        if y == util.GENRE_LABELS[genre]:
            true.append(genre)
        else:
            true.append(f"not {genre}")

    return conf.ConfusionMatrix([f"not {genre}", genre], true, guessed)

def get_matrices(partition, allowed_features):
    '''
    Converts a partition to a X matrix of features and a y vector of labels
    partition: the partition object to be converted
    allowed_features: the features allowed in X
    return a tuple of X,y
    '''
    X = np.zeros([partition.n, len(allowed_features)])
    y = np.zeros([partition.n])
    for i in range(partition.n):
        y[i] = partition.data[i].label
        j = 0
        for feature in allowed_features:
            X[i][j] = partition.data[i].features[feature]
            j += 1

    normalize(X) #in this specific application, the numbers are large, so we normalzie X
    X = add_ones(X) #adding ones for modeling purposes

    return X,y


def normalize(matrix):
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


def add_ones(matrix):
    '''
    Concatenates a column of ones to the start of matrix
    matrix: the numpy matrix to add ones to
    returns the numpy matrix with the column of ones
    '''
    ones = np.ones((len(matrix), 1))
    new = np.concatenate((ones, matrix), 1)
    return new


if __name__ == "__main__":
    main()
