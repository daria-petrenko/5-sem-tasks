import numpy as np
from scipy.signal import convolve2d


def calc_expectations(h, w, X, Q):
    conv_matr = np.ones((h, w))
    Q = convolve2d(Q, conv_matr, mode='full')
    Q = Q[:np.size(X, 0), :np.size(X, 1)]
    return X * Q
