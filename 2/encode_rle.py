import numpy as np


def encode_rle(x):
    y = np.hstack((np.ones(1), x[:np.size(x) - 1]))
    first_positions = x != y
    first_positions[0] = True
    indexes_1 = np.arange(np.size(x))[first_positions]
    indexes_2 = np.hstack((indexes_1[1:], np.array([np.size(x)])))
    return x[first_positions], indexes_2 - indexes_1
