from Partition import Partition, Example
import csv

GENRES = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae', 'rock']

GENRE_LABELS = {}
for i in range(len(GENRES)):
    GENRE_LABELS[GENRES[i]] = i

def main():
    partition = read_csv("data/sample_3_sec.csv")
    print(partition.F)

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
            if key not in ['filename', 'length']:
                keys.append(key)
        break

    vals = [[] for i in range(len(keys))] #one list for each feature

    for row in csv_file: #add values to each list (feature) in vals
        for i in range(2, len(row)):
            if row[i] not in vals[i - 2]: #no duplicates
                vals[i - 2].append(row[i])


    csv_file = csv.reader(open(filename, 'r'), delimiter=',') #reopen file
    for row in csv_file: #skip the first row
        break

    for row in csv_file:
        ex_features = {}
        ex_label = 0
        for i in range(len(row)-2): #here we hardcode the example for this lab
            for genre in GENRES:
                if genre in row[i]:
                    ex_label = GENRE_LABELS[genre]
            else:
                ex_features[keys[i]] = row[i]
        data.append(Example(ex_features, ex_label))

    for i in range(len(keys)): #constructs the dictionary
        F[keys[i]] = vals[i]

    return Partition(data, F, len(GENRES)) #hardcode number of labels for this lab

if __name__ == '__main__':
    main()
