import pickle

import numpy as np


def load_data_cifar(train_file, test_file):
    """
    Load training and test data and preprocess for training and evaluation.

    Args:
        train_file (str): path to file containing training data
        test_file (str): path to file containing test data
    """

    # Function for unpickling into dictionary.
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    # Load training and test data dictionaries.
    train_dict = unpickle(train_file)
    test_dict = unpickle(test_file)

    # Extract training data and class values.
    train_data = np.array(train_dict['data']) / 255.0
    train_class = np.array(train_dict['labels'])

    # Encode classes using one-hot encoding. Each row represents an ideal neural network output.
    train_class_one_hot = np.zeros((train_data.shape[0], 10))
    train_class_one_hot[np.arange(train_class.shape[0]), train_class] = 1.0

    # Extract training data and class values.
    test_data = np.array(test_dict['data']) / 255.0
    test_class = np.array(test_dict['labels'])

    # Encode classes using one-hot encoding. Each row represents an ideal neural network output.
    test_class_one_hot = np.zeros((test_class.shape[0], 10))
    test_class_one_hot[np.arange(test_class.shape[0]), test_class] = 1.0

    # Return transposed training and test data and class matrices.
    return train_data.transpose(), train_class_one_hot.transpose(), test_data.transpose(), test_class_one_hot.transpose()
