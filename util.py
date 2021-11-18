from Partition import Partition, Example
import csv

GENRES = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae', 'rock']

GENRE_LABELS = {}
for i in range(len(GENRES)):
    GENRES_LABELS[GENRES[i]] = i

def read_csv(filename):
    """
    Borrowed from Lab 5
    reads a csv file into a partition
    filename: the string filename for the csv file
    return the partition made from the csv
    """
    csv_file = csv.reader(open(filename, 'r'), delimiter=',')
    F = {}
    keys = []
    data = []

    for row in csv_file: #get feature names
        for key in row:
            keys.append(key)
        break

    vals = [[] for i in range(len(keys))] #one list for each feature

    for row in csv_file: #add values to each list (feature) in vals
        for i in range(len(row)):
            if row[i] not in vals[i]: #no duplicates
                vals[i].append(row[i])


    csv_file = csv.reader(open(filename, 'r'), delimiter=',') #reopen file
    for row in csv_file: #skip the first row
        break

    for row in csv_file:
        ex_features = {}
        ex_label = 0
        for i in range(len(row)): #here we hardcode the example for this lab
            for genre in GENRES:
                if genre in row[i]:
                    ex_label = GENRE_LABELS[row[i]]
            else:
                ex_features[keys[i]] = row[i]
        data.append(Example(ex_features, ex_label))

    for i in range(len(keys)): #constructs the dictionary
        F[keys[i]] = vals[i]

    return Partition(data, F, len(GENRES)) #hardcode number of labels for this lab
