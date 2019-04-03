import numpy as np


def euclidean_distance(X, Y):
    return np.sqrt(
            np.sum(X**2, axis=1)[:, np.newaxis] +
            np.sum(Y**2, axis=1) - 2 * X.dot(Y.T)
            )


def cosine_distance(X, Y):
    X = X.astype('float64') / np.sqrt(np.sum(X**2, axis=1)[:, np.newaxis])
    Y = Y.astype('float64') / np.sqrt(np.sum(Y**2, axis=1)[:, np.newaxis])
    return 1 - X.dot(Y.T)
