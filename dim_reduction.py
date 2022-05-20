"""
CS260 Project
PCA, t-SNE visualizations
Author: Keith Mburu
Author: Matt Gusdorff
Date: 12/17/2021
"""

import util
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
from sklearn.manifold import TSNE

def setup(type, filename, partition, cont_to_disc):
    """
    Setting up the model and fitting it on the data
    """
    if os.path.exists(f"objects/{type}_{filename}.pkl"):
        print(f"\nloading {type}\n")
        model_x = util.object_load(f"{type}_" + filename)
        print("\nloading X, y\n")
        X, y = util.object_load("X_y_" + filename)
    else:   
        if os.path.exists("objects/" + "X_y_" + filename + ".pkl"):
            print("\nloading X, y\n")
            X, y = util.object_load("X_y_" + filename)
        else:
            print("\ncreating X, y\n")
            X = np.zeros([partition.n, len(partition.F)])
            y = np.zeros([partition.n, 1])
            for i, example in enumerate(partition.data):
                features = []
                for key in example.features:
                    if cont_to_disc:
                        features.append(1 if example.features[key] == "True" else 0)
                    else:
                        features.append(example.features[key])
                X[i] = features
                y[i] = example.label
            print("\nstoring X, y\n")
            util.object_store([X, y], "X_y_" + filename)
        if type == "pca":
            model = PCA(n_components = 2).fit(X)
        else:
            model = TSNE(n_components=2, learning_rate='auto', init='random').fix(X)
        model_x = model.transform(X)
        print(f"\nstoring {type}\n")
        util.object_store(model_x, f"{type}_" + filename)
    return model_x, y

def plot(type, model, cont_to_disc, y, n):
    """
    Plotting result
    """
    # setting up plot colors
    color_options = ['blue', 'orange', 'red', 'purple', 'cyan', 'plum',
                    'yellow', 'lime', "indigo", "teal"]
    color_dict = {}
    colors = []
    j = 0
    for i in range(len(y)):
        if y[i, 0] in color_dict:
            colors.append(color_dict[y[i, 0]])
        else:
            color = color_options[j]
            j += 1
            color_dict[y[i, 0]] = color
            colors.append(color)
    # plotting
    plt.scatter(model[:, 0], model[:, 1], s=2, c=colors)
    plt.title(f"GTZAN Dataset {type.upper()}")
    x_label = "PC1" if type == "pca" else "Dimension 1"
    y_label = "PC2" if type == "pca" else "Dimension 2"
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # # creating legend
    # leg_objects = []
    # for i in range(len(color_dict)):
    #     circle, = plt.plot([], 'o', c=color_dict[i])
    #     leg_objects.append(circle)
    # plt.legend(leg_objects, util.GENRES)
    plt.savefig(f"figures/{type}_{cont_to_disc}_n{n}.png")
    plt.show()
    plt.clf()
