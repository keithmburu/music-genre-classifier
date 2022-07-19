"""
CS260 Project
Utility contains methods for data preprocessing and data persistence
Author: Keith Mburu
Author: Matt Gusdorff
Date: 12/17/2021
"""

from Partition import Partition, Example
import csv
import cont_to_disc
import _pickle as pkl
import os.path
import math

GENRES = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae', 'rock']

GENRE_LABELS = {'blues': 0,'classical': 1,'country': 2,'disco': 3,'hiphop': 4,
                'jazz': 5,'metal': 6,'pop': 7,'reggae': 8, 'rock': 9}

GENRE_LABELS_R = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop',
                 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}

SORTED_CONTINUOUS_FEATURES =['tempo', 'spectral_centroid_var', 'perceptr_var', 'rms_var', 'rolloff_var', 'mfcc1_mean', 'rolloff_mean',
                             'spectral_bandwidth_mean', 'chroma_stft_mean', 'spectral_centroid_mean', 'rms_mean', 'spectral_bandwidth_var', 'mfcc1_var', 'mfcc4_var',
                             'mfcc2_mean', 'mfcc6_var', 'zero_crossing_rate_var', 'mfcc3_var', 'mfcc7_var', 'mfcc5_var', 'mfcc4_mean', 'harmony_var', 'mfcc2_var',
                             'zero_crossing_rate_mean', 'mfcc9_mean', 'mfcc6_mean', 'mfcc8_mean', 'mfcc13_mean', 'mfcc3_mean', 'mfcc8_var', 'chroma_stft_var',
                             'mfcc7_mean', 'mfcc12_mean', 'mfcc15_mean', 'mfcc9_var', 'mfcc10_var', 'mfcc12_var', 'mfcc10_mean', 'mfcc17_mean', 'mfcc5_mean',
                             'mfcc11_mean', 'harmony_mean', 'mfcc11_var', 'perceptr_mean', 'mfcc20_var', 'mfcc16_mean', 'mfcc13_var', 'mfcc19_mean', 'mfcc15_var',
                             'mfcc19_var', 'mfcc14_var', 'mfcc18_var', 'mfcc14_mean', 'mfcc16_var', 'mfcc17_var', 'mfcc18_mean', 'mfcc20_mean']

for i in range(len(GENRES)):
    GENRE_LABELS[GENRES[i]] = i

def read_csv(filename, convert_to_discrete, percentage_of_features=100):
    """
    Borrowed from Lab 5
    reads a csv file into a partition
    filename: the string filename for the csv file
    return the partition made from the csv
    """
    partition = Partition([], {}, 0)
    if os.path.exists("objects/" + str(convert_to_discrete) + "_" + str(percentage_of_features) + "%_" + filename[5:] + ".pkl"):
        print("\nloading partition\n")
        partition = object_load(str(convert_to_discrete) + "_" + str(percentage_of_features) + "%_" + filename[5:])
    else:
        print("\nreading csv\n")
        csv_file = csv.reader(open(filename, 'r'), delimiter=',')
        F = {}
        keys = []
        data = []

        for row in csv_file: #get feature names
            for key in row:
                if key not in ['filename', 'length', 'label']:
                    keys.append(key)
            break

        vals = [[] for i in range(len(keys))] #one list for each feature

        for row in csv_file: #add values to each list (feature) in vals
            for i in range(2, len(row)-1):
                if row[i] not in vals[i - 2]: #no duplicates
                    vals[i - 2].append(row[i])


        csv_file = csv.reader(open(filename, 'r'), delimiter=',') #reopen file
        for row in csv_file: #skip the first row
            break

        for row in csv_file:
            ex_features = {}
            ex_label = -1
            for i in range(len(row)-1): #here we hardcode the example for this lab
                if i == 0:
                    genre = row[i].split('.')[0]
                    ex_label = GENRE_LABELS[genre]
                elif i != 1:
                    ex_features[keys[i-2]] = row[i]
            data.append(Example(ex_features, ex_label))

        for i in range(len(keys)): #constructs the dictionary
            F[keys[i]] = vals[i]

        partition_temp = Partition(data, F, len(GENRES))
        best_features = SORTED_CONTINUOUS_FEATURES[:math.ceil((percentage_of_features/100)*len(partition_temp.F))]
        for example in data:
            for feature in example.features:
                if feature not in best_features:
                    del feature
        for feature in F:
            if feature not in best_features:
                del feature

        F_disc = {}
        if convert_to_discrete:
            print("\nconverting cont to disc\n")
            # converting continuous features to discrete
            for feature in F:
                cont_to_disc.convert_feature(feature, data, F_disc)
                # print(F_disc, "\n")
        else:
            F_disc = F
        # print(F_disc)

        partition = Partition(data, F_disc, len(GENRES)) #hardcode number of labels for this lab
        print("storing partition")
        object_store(partition, str(convert_to_discrete) + "_" + str(percentage_of_features) + "%_" + filename[5:])
    # print(partition.F)
    # return partition
    return str(convert_to_discrete) + "_" + str(percentage_of_features) + "%_" + filename[5:], partition

def object_store(object, file_name):
    with open("objects/" + file_name + ".pkl", 'wb') as file:
        pkl.dump(object, file)

def object_load(file_name):
    with open("objects/" + file_name + ".pkl", 'rb') as file:
        return pkl.load(file)
