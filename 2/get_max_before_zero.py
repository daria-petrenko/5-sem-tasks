import numpy as np


def get_max_before_zero(x):
    y = np.ones((1))
    y = np.hstack((y, x))
    matr = (y == 0)[:np.size(y) - 1]
    if np.size(x[matr]) == 0:
        return None
    else:
        return np.max(x[matr])
