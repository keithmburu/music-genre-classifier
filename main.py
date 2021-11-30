import LogisticRegression as lr

import math
import util

def main():
    train_partition = util.read_csv("data/features_n800_30_sec_shuffled.csv", False)
    test_partition = util.read_csv("data/features_n200_30_sec_shuffled.csv", False)
    lr_generator = lr.LogisticRegression(train_partition)

    for genre in util.GENRES:
        print(f"{genre} accuracy:",genre_accuracy(lr_generator, genre, test_partition))

def genre_accuracy(lr_generator, genre, partition):
    model = lr_generator.model(genre)
    correct = 0

    for i in range(partition.n):
        if round(model(lr_generator.X[i]), 0) == 1 and partition.data[i].label == util.GENRE_LABELS[genre]:
            correct += 1
        elif round(model(lr_generator.X[i]), 0) == 0 and partition.data[i].label != util.GENRE_LABELS[genre]:
            correct += 1

    return correct/partition.n























if __name__ == "__main__":
    main()