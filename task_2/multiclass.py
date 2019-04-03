import numpy as np
from scipy import sparse


from optimization import GDClassifier, SGDClassifier


class MulticlassStrategy:   
    def __init__(self, classifier, mode, **kwargs):
        self.classifier = classifier
        self.mode = mode
        self.kwargs = kwargs
        pass
        
        
    def fit(self, X, y):
        self.num_classes = np.size(np.unique(y))
        if(self.mode == 'one_vs_all'):
            self.w = np.zeros((self.num_classes, X.shape[1]))
            for i in range(self.num_classes):
                mask = y == i
                curr_y = 2 * mask - 1 # 1 if y == i, -1 if not
                curr_lr = self.classifier(**self.kwargs)
                curr_lr.fit(X, curr_y)
                self.w[i] = curr_lr.get_weights()
        elif(self.mode == 'all_vs_all'):
            self.w = np.zeros((self.num_classes * (self.num_classes - 1) // 2, 
                              X.shape[1]))
            num = 0
            i_list = []
            j_list = []
            for i in range(self.num_classes - 1):
                for j in range(i + 1, self.num_classes):
                    i_list.append(i)
                    j_list.append(j)
                    matr_i = y == i
                    matr_j = y == j
                    matr = np.logical_or(matr_i, matr_j)
                    curr_x = X[matr]
                    curr_y = y.copy()
                    curr_y[matr_i] = 1
                    curr_y[matr_j] = -1
                    curr_y = curr_y[matr]
                    curr_lr = self.classifier(**self.kwargs)
                    curr_lr.fit(curr_x, curr_y)
                    self.w[num] = curr_lr.get_weights() 
                    num += 1
            self.i_arr = np.array(i_list)
            self.j_arr = np.array(j_list)
        
    def predict(self, X):
        if(self.mode == 'one_vs_all'):
            if(sparse.issparse(X)):
                return np.argmax(np.asarray(X.dot(self.w.T)), axis=1)
            else:
                return np.argmax(np.dot(X, self.w.T), axis=1)
        elif(self.mode == 'all_vs_all'):
            if(sparse.issparse(X)):
                pred = np.asarray(X.dot(self.w.T))
            else:
                pred = np.dot(X, self.w.T)
            mask = pred > 0
            self.i_arr = np.broadcast_to(self.i_arr, pred.shape)
            self.j_arr = np.broadcast_to(self.j_arr, pred.shape)
            pred[mask] = self.i_arr[mask]                            
            mask = np.logical_not(mask)
            pred[mask] = self.j_arr[mask]
            return np.argmax(np.apply_along_axis(
                lambda a:np.bincount(a, minlength=self.num_classes), 
                -1, pred.astype(int)), axis=1)