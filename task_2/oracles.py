import numpy as np
from scipy import sparse
from scipy.special import logsumexp, expit


class BaseSmoothOracle:

    def func(self, w):
        if(sparse.issparse(self.X)):
            if(w.ndim == 1):
                return np.sum(logsumexp(
                    np.vstack((
                            -1 * np.asarray(self.X.dot(w)) * self.y,
                            np.zeros(np.size(self.y))
                            )).T,
                    axis=1
                    )) / np.size(self.y) + self.l2_coef * np.dot(w, w) / 2
            else:
                max_val = np.amax(np.asarray(self.X.dot(w.T)), axis=1)
                return -1 * np.sum(
                    np.squeeze(np.asarray(
                            self.X.multiply(w[self.y.astype(int)]).sum(axis=1)
                            ), axis=1) -
                    max_val -
                    logsumexp(
                            np.asarray(self.X.dot(w.T)) -
                            max_val[:, np.newaxis],
                            axis=1)
                    ) / np.size(self.y) + self.l2_coef * np.sum(w * w) / 2
        else:
            if(w.ndim == 1):
                return np.sum(logsumexp(
                        np.vstack((
                                -1 * np.dot(self.X, w) * self.y,
                                np.zeros(np.size(self.y))
                                )).T,
                        axis=1
                        )) / np.size(self.y) + self.l2_coef * np.dot(w, w) / 2
            else:
                max_val = np.amax(np.dot(self.X, w.T), axis=1)
                return -1 * np.sum(
                        np.sum(self.X * w[self.y.astype(int)], axis=1) - 
                        max_val -
                        logsumexp(
                                np.dot(self.X, w.T) -
                                max_val[:, np.newaxis],
                                axis=1)
                        ) / np.size(self.y) + self.l2_coef * np.sum(w * w) / 2

    def grad(self, w):
        if(self.X.dtype == int):
            min_val = np.iinfo(self.X.dtype).min
            max_val = np.iinfo(self.X.dtype).max
        elif(self.X.dtype == float):
            min_val = np.finfo(self.X.dtype).min
            max_val = np.finfo(self.X.dtype).max
        if(sparse.issparse(self.X)):
            if(w.ndim == 1):
                arg = np.asarray(self.X.dot(w)) * self.y
                return np.squeeze(np.asarray(
                    self.X.multiply(self.y[:, np.newaxis]).multiply(
                        -1 * (np.clip(
                                    np.exp(-1 * arg),
                                    min_val, max_val
                            ) * expit(arg))[:, np.newaxis]
                    ).sum(axis=0)), axis=0) / \
                    np.size(self.y) + self.l2_coef * w
            else:
                mask = np.arange(np.size(w, 0))
                mask = mask[:, np.newaxis] == self.y[np.newaxis, :]
                max_arg = np.amax(np.asarray(self.X.dot(w.T)), axis=1)
                arg = np.asarray(self.X.dot(w.T)) - max_arg[:, np.newaxis]
                return self.X.transpose().dot( 
                    -1 * mask.T +
                    np.clip(np.exp(arg), min_val, max_val) /
                    np.clip(
                        np.sum(np.exp(arg), axis=1),
                        min_val, max_val)[:, np.newaxis]
                ).T / np.size(self.y) + self.l2_coef * w
        else:
            if(w.ndim == 1):
                arg = np.dot(self.X, w) * self.y
                return np.sum(
                        -1 * (np.clip(
                                np.exp(-1 * arg),
                                min_val, max_val
                        ) * expit(arg))[:, np.newaxis] *
                        self.X * self.y[:, np.newaxis],
                        axis=0
                        ) / np.size(self.y) + self.l2_coef * w
            else:
                mask = np.arange(np.size(w, 0))
                mask = mask[:, np.newaxis] == self.y[np.newaxis, :]
                max_arg = np.amax(np.dot(self.X, w.T), axis=1)
                arg = np.dot(self.X, w.T) - max_arg[:, np.newaxis]
                return np.dot(
                    self.X.T, 
                    -1 * mask.T + 
                    np.clip(np.exp(arg), min_val, max_val) /
                    np.clip(
                        np.sum(np.exp(arg), axis=1),
                        min_val, max_val
                    )[:, np.newaxis]).T / np.size(self.y) + self.l2_coef * w

class BinaryLogistic(BaseSmoothOracle):

    def __init__(self, l2_coef):
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        self.X = X
        self.y = y
        return super().func(w)

    def grad(self, X, y, w):
        self.X = X
        self.y = y
        return super().grad(w)


class MulticlassLogistic(BaseSmoothOracle):

    def __init__(self, l2_coef):
        self.l2_coef = l2_coef

    def func(self, X, y, w):
        self.X = X
        self.y = y
        return super().func(w)

    def grad(self, X, y, w):
        self.X = X
        self.y = y
        return super().grad(w)
