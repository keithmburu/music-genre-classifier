"""
CS260 Project
Converting continuous train and test features to discrete
Author: Keith Mburu
Author: Matt Gusdorff
Date: 12/17/2021
"""

import os
import util

def convert_feature(f, data, F_disc):
    """
    Convert one feature (name f) from continuous to discrete.
    Credit: adapted from lab 6, based on original code from Ameet Soni.
    """

    # first combine the feature values (for f) and the labels
    combineXy = []
    for example in data:
        combineXy.append([example.features[f],example.label])
    combineXy.sort(key=lambda elem: elem[0]) # sort by feature

    # first need to merge uniques
    unique = []
    u_label = {}
    for elem in combineXy:
        if elem[0] not in unique:
            unique.append(elem[0])
            u_label[elem[0]] = elem[1]
        else:
            if u_label[elem[0]] != elem[1]:
                u_label[elem[0]] = None

    # find switch points (label changes)
    switch_points = []
    for j in range(len(unique)-1):
        # print(u_label[unique[j]])
        if u_label[unique[j]] != u_label[unique[j+1]] or u_label[unique[j]] \
            == None:
            switch_points.append((float(unique[j])+float(unique[j+1]))/2) # midpoint

    # add a feature for each switch point (keep feature vals as strings)
    for s in switch_points:
        key = f+"<="+str(s)
        for i in range(len(data)):
            if float(data[i].features[f]) <= s:
                data[i].features[key] = "True"
            else:
                data[i].features[key] = "False"
        # print(key)
        F_disc[key] = ["False", "True"]
        # print(F_disc[key])

    # delete this feature from all the examples
    for example in data:
        del example.features[f]

def refeaturize(train_partition, test_partition, test_file):
    """
    Replaces the continuous features of each test example with the 
    discrete features of the train examples
    """
    test_partition_refeaturized = test_partition
    if os.path.exists("objects/refeaturized_" + test_file + ".pkl"):
        print("\nloading refeaturized\n", )
        test_partition_refeaturized = util.object_load("refeaturized_" + test_file)
    else:
        # refeaturizing test examples
        test_partition_refeaturized.F = train_partition.F.copy()
        discrete_features = list(train_partition.F.keys())
        print("\nrefeaturizing\n")
        for example in test_partition_refeaturized.data:
            j = 0
            done = 0
            old_features = list(example.features.keys())
            for i in range(len(old_features)):
                old_feature = old_features[i]
                feature_value = example.features[old_feature]
                while old_feature in discrete_features[j] and not done:
                    new_feature = discrete_features[j]
                    disc_feature_threshold = discrete_features[j].replace(old_feature + "<=", "")
                    if feature_value <= disc_feature_threshold:
                        example.features[new_feature] = "True"
                    else:
                        example.features[new_feature] = "False"
                    if j+1 < len(discrete_features):
                        j += 1
                    else:
                        done = 1
                del example.features[old_feature]
        print("\nstoring refeaturized\n")
        util.object_store(test_partition_refeaturized, "refeaturized_" + test_file)
    return test_partition_refeaturized
