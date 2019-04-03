import numpy as np
from sklearn.neighbors import NearestNeighbors

from distances import euclidean_distance, cosine_distance


class KNNClassifier:
    def __init__(self, k, strategy, metric, 
            weights, test_block_size):
        self.k = k
        self.strategy = strategy
        self.metric = metric
        self.weights = weights
        self.test_block_size = test_block_size

    def fit(self, X, y):
        if(self.strategy == 'my_own'):
            self.X = X
            self.y = y
        else:
            self.y = y
            self.sklearn_knn = NearestNeighbors(
                    n_neighbors=self.k, algorithm=self.strategy,
                    metric=self.metric).fit(X)

    def find_kneighbors(self, X, return_distance):
        iter_num = np.size(X, 0) // self.test_block_size
        if(np.size(X, 0) % self.test_block_size):
            iter_num += 1
        ind_list = list()
        if(return_distance):
            dist_list = list()
        for i in range(iter_num):
            ind_beg = self.test_block_size * i
            ind_end = self.test_block_size * (i + 1)
            if(self.strategy == 'my_own'):
                if(self.metric == 'euclidean'):
                    distance_matrix = euclidean_distance(X[ind_beg:ind_end], self.X)
                elif(self.metric == 'cosine'):
                    distance_matrix = cosine_distance(X[ind_beg:ind_end], self.X)
                k_ind_matrix = np.argsort(distance_matrix, axis=1)[:, :self.k]
                ind_list.append(k_ind_matrix)
                if(return_distance):
                    dist_list.append(distance_matrix[
                        np.arange(np.size(k_ind_matrix, 0))[:, np.newaxis],
                        k_ind_matrix])
            else:
                if(return_distance):
                    dist, ind = self.sklearn_knn.kneighbors(
                    X[ind_beg:ind_end], n_neighbors=self.k,
                    return_distance=return_distance)
                    dist_list.append(dist)
                    ind_list.append(ind)
                else:
                    ind_list.append(
                        self.sklearn_knn.kneighbors(
                                X[ind_beg:ind_end], n_neighbors=self.k,
                                return_distance=return_distance))
        if(return_distance):
            return np.concatenate(dist_list), np.concatenate(ind_list)
        else:
            return np.concatenate(ind_list)

    def predict(self, X):
        if(not self.weights):
            k_ind_matrix = self.find_kneighbors(X, False)
            k_y = self.y[k_ind_matrix]
            result_matrix = np.empty(np.size(X, 0))
            for j in range(np.size(X, 0)):
                result_matrix[j] = np.argmax(np.bincount(k_y[j]))
            return result_matrix
        else:
            distance_matrix, k_ind_matrix = self.find_kneighbors(X, True)
            k_y = self.y[k_ind_matrix]
            result_matrix = np.empty(np.size(X, 0))
            for j in range(np.size(X, 0)):
                result_matrix[j] = np.argmax(
                        np.bincount(
                                k_y[j],
                                weights=1 / (distance_matrix[j] + 1e-5)
                                )
                        )
            return result_matrix
