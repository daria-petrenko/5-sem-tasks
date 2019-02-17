import numpy as np


def get_nonzero_diag_product(X):
    min_axis = min(np.size(X, 0), np.size(X, 1))
    X = X[:min_axis, :min_axis]
    E = np.eye(np.size(X, 0))
    y = np.ones(np.size(X, 0))
    x = np.dot(X * E, y)
    x = x[x.astype(bool)]
    if(np.size(x) == 0):
        return None
    else:
        return x.prod()
