import numpy as np

from nearest_neighbors import KNNClassifier

def kfold(n, n_folds):
    ind_array = np.arange(n)
    np.random.shuffle(ind_array)
    fold_size = n // n_folds
    left = n - fold_size * n_folds
    folds_list = list()
    for i in range(left):
        beg_ind = (fold_size + 1) * i
        end_ind = (fold_size + 1) * (i + 1)
        folds_list.append((
                np.hstack((ind_array[:beg_ind], ind_array[end_ind:])),
                ind_array[beg_ind: end_ind]))
    for i in range(n_folds - left):
        beg_ind = (fold_size + 1) * left + fold_size * i
        end_ind = (fold_size + 1) * left + fold_size * (i + 1)
        folds_list.append((
                np.hstack((ind_array[:beg_ind], ind_array[end_ind:])),
                ind_array[beg_ind: end_ind]))
    return folds_list


def knn_cross_val_score(X, y, k_list, score, cv, **kwargs):
    if (cv is None):
        cv = kfold(np.size(X, 0), 3)
    score_dict = dict()
    curr_ind = list()
    weight_flag = kwargs['weights']
    if(weight_flag):
        curr_weights = list()
    curr_score = np.empty(len(cv))
    for j in range(len(cv)):
        classifier = KNNClassifier(k_list[len(k_list) - 1], **kwargs)
        classifier.fit(X[cv[j][0]], y[cv[j][0]])
        y_knn = classifier.predict(X[cv[j][1]])
        if(weight_flag):
            curr_w, curr_i = classifier.find_kneighbors(X[cv[j][1]], True)
            curr_weights.append(curr_w)
            curr_ind.append(curr_i)
        else:
            curr_i = classifier.find_kneighbors(X[cv[j][1]], False)
            curr_ind.append(curr_i)
        if(score == 'accuracy'):
            num_diff = np.sum(y_knn == y[cv[j][1]])
            curr_score[j] = num_diff / np.size(y_knn)
    score_dict[k_list[len(k_list) - 1]] = curr_score
    for i in range(len(k_list) - 2, -1, -1):
        k_diff = k_list[i + 1] - k_list[i]
        curr_score = np.empty(len(cv))
        for j in range(len(cv)):
            curr_ind[j] = curr_ind[j][:, :-k_diff]
            if(weight_flag):
                curr_weights[j] = curr_weights[j][:, :-k_diff]
                y_ind = y[cv[j][0]][curr_ind[j]]
                y_knn = np.empty(np.size(y_ind, 0))
                for k in range(np.size(y_ind, 0)):
                    y_knn[k] = np.argmax(
                            np.bincount(
                                    y_ind[k],
                                    weights=1 / (curr_weights[j][k] + 1e-5)
                                    )
                            )
            else:
                y_ind = y[cv[j][0]][curr_ind[j]]
                y_knn = np.empty(np.size(y_ind, 0))
                for k in range(np.size(y_ind, 0)):
                    y_knn[k] = np.argmax(np.bincount(y_ind[k]))
            if(score == 'accuracy'):
                num_diff = np.sum(y_knn == y[cv[j][1]])
                curr_score[j] = num_diff / np.size(y_knn)
        score_dict[k_list[i]] = curr_score
    return score_dict
