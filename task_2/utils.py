import numpy as np

def grad_finite_diff(function, w, eps=1e-8):
    I = np.eye(np.size(w))
    return (np.apply_along_axis(function, axis=1, arr=w[:, np.newaxis] + eps * I) - 
            function(w)[:, np.newaxis]).sum(axis=1) / eps
