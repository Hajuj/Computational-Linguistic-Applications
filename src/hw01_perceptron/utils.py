import random
from nltk import word_tokenize


def dot(dictA, dictB):
    return sum(dictA[key]*dictB.get(key, 0) for key in dictA)


def normalized_tokens(text):
    return [token.lower() for token in word_tokenize(text)]


class DataInstance:
    def __init__(self, feature_counts, label):
        """ A data instance consists of a dictionary with feature counts (string -> int) and a label (True or False)."""
        self.feature_counts = feature_counts
        self.label = label

    @classmethod
    def from_list_of_feature_occurrences(cls, feature_list, label):
        """ Creates feature counts for all features in the list."""
        feature_counts = dict()
        for occ in feature_list:
            feature_counts[occ] = feature_counts.get(occ,0)+1
        return cls(feature_counts, label)

    @classmethod
    def from_text_file(cls, filename, label):
        with open(filename, 'r') as myfile:
            token_list = normalized_tokens(myfile.read().strip())
        return cls.from_list_of_feature_occurrences(token_list, label)


class Dataset:
    def __init__(self, instance_list):
        """ A data set is defined by a list of instances """
        self.instance_list = instance_list
        self.feature_set = set.union(*[set(inst.feature_counts.keys()) for inst in instance_list])

    def get_topn_features(self, n):
        """ This returns a set with the n most frequently occurring features (i.e. the features that are contained in 
        most instances). """
        return set()  # TODO: Ex. 4: Return set of features that occur in most instances.

    def set_feature_set(self, feature_set):
        """
        This restricts the feature set. Only features in the specified set all retained. All other feature are removed
        from all instances in the dataset AND from the feature set."""
        # TODO: Ex. 5: Filter features according to feature set.
        pass

    def most_frequent_sense_accuracy(self):
        """ Computes the accuracy of always predicting the overall most frequent sense for all instances in the 
        dataset. """
        return 0.0  # TODO: Ex. 6: Return accuracy of always predicting most frequent label in data set.

    def shuffle(self):
        """ Shuffles the dataset. Beneficial for some learning algorithms."""
        random.shuffle(self.instance_list)
