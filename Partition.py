#borrows from lab 5
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
