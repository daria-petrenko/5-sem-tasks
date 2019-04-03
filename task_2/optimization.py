import numpy as np
import time
from scipy import sparse
from scipy.special import expit

from oracles import BinaryLogistic, MulticlassLogistic


class GDClassifier:

    def __init__(self, loss_function, step_alpha=0.1, step_beta=1,
                 tolerance=1e-5, max_iter=1000, **kwargs):
        self.loss_function = loss_function
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.kwargs = kwargs

    def fit(self, X, y, X_test=np.zeros(1), y_test=np.zeros(1), 
            w_0=None, trace=False):
        if(self.loss_function == 'binary_logistic'):
            if(w_0 is None):
                w_0 = np.zeros(np.size(X, 1))
            self.lr = BinaryLogistic(**self.kwargs)
        elif(self.loss_function == 'multinomial_logistic'):
            if(w_0 is None):
                w_0 = np.zeros((np.size(np.unique(y)), np.size(X, 1)))
            self.lr = MulticlassLogistic(**self.kwargs)
        self.w = w_0.copy()
        last_func = self.lr.func(X, y, self.w)
        curr_func = last_func
        if(trace):
            self.history = dict()
            self.history['time'] = [0.0]
            self.history['func'] = [last_func]
            self.history['acc'] = [np.sum(np.equal(
                    y_test, self.predict(X_test))) / np.size(y_test)]
            start = time.time()
        num_iter = 0
        while(num_iter == 0 or
              (np.abs(curr_func - last_func) >= self.tolerance and
               num_iter < self.max_iter)):
            num_iter += 1
            self.w -= self.lr.grad(X, y, self.w) * \
                self.step_alpha / num_iter ** self.step_beta
            last_func = curr_func
            curr_func = self.lr.func(X, y, self.w)
            if(trace):
                end = time.time()
                self.history['time'].append(end - start)
                self.history['func'].append(curr_func)
                self.history['acc'].append(np.sum(np.equal(y_test, 
                            self.predict(X_test))) / np.size(y_test))
        if(trace):
            return self.history

    def predict(self, X):
        if(self.loss_function == 'binary_logistic'):
            if(sparse.issparse(X)):
                return np.sign(np.asarray(X.dot(self.w.T)))
            else:
                return np.sign(np.dot(X, self.w.T))
        else:
            if(sparse.issparse(X)):
                return np.argmax(np.asarray(X.dot(self.w.T)), axis=1)
            else:
                return np.argmax(np.dot(X, self.w.T), axis=1)

    def predict_proba(self, X):
        if(self.loss_function == 'binary_logistic'):
            if(sparse.issparse(X)):
                return expit(np.asarray(X.dot(self.w.T)))
            else:
                return expit(np.dot(X, self.w.T))
        elif(self.loss_function == 'multinomial_logistic'):
            if(sparse.issparse(X)):
                softmax = np.exp(np.asarray(X.dot(self.w.T)) -
                                 np.amax(
                                         np.asarray(X.dot(self.w.T)),
                                         axis=1
                                         )[:, np.newaxis]
                                 )
            else:
                softmax = np.exp(np.dot(X, self.w.T) -
                                 np.amax(
                                         np.dot(X, self.w.T),
                                         axis=1
                                         )[:, np.newaxis]
                                 )
            return softmax / np.sum(softmax, axis=1)[:, np.newaxis]

    def get_objective(self, X, y):
        return self.lr.func(X, y, self.w)

    def get_gradient(self, X, y):
        return self.lr.grad(X, y, self.w)

    def get_weights(self):
        return self.w


class SGDClassifier(GDClassifier):

    def __init__(self, loss_function, batch_size, step_alpha=1, step_beta=0,
                 tolerance=1e-5, max_iter=100000, random_seed=153, **kwargs):
        GDClassifier.__init__(self, loss_function=loss_function, 
                              step_alpha=step_alpha, step_beta=step_beta,
                             tolerance=tolerance, max_iter=max_iter, **kwargs)
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.kwargs = kwargs

    def fit(self, X, y, X_test, y_test, w_0=None, trace=False, log_freq=1):
        np.random.seed(self.random_seed)
        if(self.loss_function == 'binary_logistic'):
            if(w_0 is None):
                w_0 = np.zeros(np.size(X, 1))
            self.lr = BinaryLogistic(**self.kwargs)
        elif(self.loss_function == 'multinomial_logistic'):
            if(w_0 is None):
                w_0 = np.zeros((np.size(np.unique(y)), np.size(X, 1)))
            self.lr = MulticlassLogistic(**self.kwargs)
        self.w = w_0.copy()
        last_func = self.lr.func(X, y, self.w)
        curr_func = last_func
        if(trace):
            self.history = dict()
            self.history['epoch_num'] = [0.0]
            self.history['time'] = [0.0]
            self.history['func'] = [last_func]
            self.history['weights_diff'] = [0.0]
            self.history['acc'] = [np.sum(np.equal(y_test, 
                        self.predict(X_test))) / np.size(y_test)]
            start = time.time()
            last_epoch_num = 0
            curr_epoch_num = 0
            last_w = self.w.copy()
            curr_w = last_w.copy()
        num_iter = 0
        ind_list = np.arange(np.size(X, 0))
        np.random.shuffle(ind_list)
        curr_ind = 0
        while(num_iter == 0 or
              (np.abs(curr_func - last_func) >= self.tolerance and
               num_iter < self.max_iter)):
            if(curr_ind >= np.size(X, 0)):
                np.random.shuffle(ind_list)
                curr_ind = 0
            num_iter += 1
            self.w -= self.lr.grad(
                    X[curr_ind:curr_ind + self.batch_size, :], 
                    y[curr_ind:curr_ind + self.batch_size], self.w
                    ) * self.step_alpha / num_iter ** self.step_beta
            last_func = curr_func.copy()
            curr_func = self.lr.func(X, y, self.w)
            if(trace):
                if(curr_ind  + self.batch_size >= np.size(ind_list)):
                    curr_epoch_num += (np.size(ind_list) - curr_ind) / \
                        np.size(ind_list)
                else:
                    curr_epoch_num += self.batch_size / np.size(ind_list)
                if(curr_epoch_num - last_epoch_num >= log_freq):
                    end = time.time()
                    last_w = curr_w.copy()
                    curr_w = self.w
                    self.history['epoch_num'].append(curr_epoch_num)
                    self.history['time'].append(end - start)
                    self.history['func'] .append(curr_func)
                    self.history['acc'].append(np.sum(np.equal(y_test, 
                                self.predict(X_test))) / np.size(y_test))
                    self.history['weights_diff'].append(
                            np.sum(
                                    (last_w - curr_w) ** 2, 
                                    axis=-1))
                    last_epoch_num = curr_epoch_num
            curr_ind += self.batch_size
        if(trace):
            return self.history
        
    def predict(self, X):
        return super().predict(X)