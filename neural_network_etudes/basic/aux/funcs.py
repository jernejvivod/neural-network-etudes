import numpy as np


def softmax(z):
    """
    The softmax function.

    Args:
        z (float): function input

    Returns:
        (float): function output

    """
    # Compute softmax value.
    exp_z = np.exp(z - np.max(z))
    return exp_z / exp_z.sum(axis=0, keepdims=True)


def cross_entropy(y, x, epsilon=1e-12):
    """
    Compute the cross-entropy between y and x.

    Args:
        y (numpy.ndarray): first vector.
        x (numpy.ndarray): second vector.

    Returns:
        (float): cross-entropy between y and x.
    """

    # Compute cross-entropy.
    targets = y.transpose()
    predictions = x.transpose()
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    return -np.sum(targets * np.log(predictions + 1e-9)) / predictions.shape[0]


def softmax_dz(output, target):
    """
    partial derivative of the cross entropy loss with respect to z at the last layer

    Args:
        output (numpy.ndarray): the output of the neural network.
        target (numpy.ndarray): the target value (ideal output).
    """

    # Compute result.
    return output - target


def sigmoid(z):
    """
    The sigmoid function

    Args:
        z (float): input

    Returns:
        (float): output
    """

    return 1.0 / (1.0 + np.exp(-z))
