import numpy as np


def replace_nan_to_means(X):
    X_copy = X.copy()
    mask = np.isnan(X_copy)  # mask of nan values
    if_all = np.all(mask, axis=0)  # columns of nan only
    number_of_not_nan = np.sum(np.logical_not(mask), axis=0)
    number_of_not_nan[if_all] = 1  # not to divide to zero
    X_copy[mask] = 0
    mean_matr = np.sum(X_copy, axis=0) / number_of_not_nan
    E = np.ones((np.size(X, 0), np.size(X, 1)))
    X_copy[mask] = (E * mean_matr)[mask]
    return X_copy
