import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize_scalar

class RandomForestMSE:
    def __init__(self, n_estimators, max_depth=None, feature_subsample_size=None, 
                 random_seed=75, **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        
        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_params = trees_parameters
        self.random_seed=random_seed
        
    def fit(self, X, y):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        y : numpy ndarray
            Array of size n_objects
        """
        np.random.seed(self.random_seed)
        self.trees_list = []
        self.ind_list = []
        if(self.feature_subsample_size is None):
            self.feature_subsample_size = np.size(X, 1) // 3  # regression task
        for i in range(self.n_estimators):
            ind = np.random.randint(np.size(X, 1), 
                                    size=self.feature_subsample_size)
            self.ind_list.append(ind)
            tree = DecisionTreeRegressor(max_depth=self.max_depth, 
                                         **self.trees_params)
            tree.fit(X[:, ind], y)
            self.trees_list.append(tree)
        
    def predict(self, X, y_val):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        self.acc_list = []
        sum = 0
        num = 0
        for tree in self.trees_list:
            sum += tree.predict(X[:, self.ind_list[num]])
            num += 1
            curr_pred = sum / num
            self.acc_list.append(np.sqrt(mean_squared_error(y_val, curr_pred)))
        return sum / self.n_estimators
    

class GradientBoostingMSE:
    def __init__(self, n_estimators, learning_rate=0.1, max_depth=5, 
                 feature_subsample_size=None, random_seed=75, **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.
        
        learning_rate : float
            Use learning_rate * gamma instead of gamma

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.
        
        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_params = trees_parameters
        self.random_seed = random_seed
        
    def fit(self, X, y):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        y : numpy ndarray
            Array of size n_objects
        """
        np.random.seed(self.random_seed)
        self.trees_list = []
        self.ind_list = []
        self.coeff_list = []
        curr_func = np.zeros(np.shape(y))
        if(self.feature_subsample_size is None):
            self.feature_subsample_size = np.size(X, 1) // 3  # regression task
        for i in range(self.n_estimators):
            ind = np.random.randint(np.size(X, 1), 
                                    size=self.feature_subsample_size)
            self.ind_list.append(ind)
            targ = - curr_func + y
            tree = DecisionTreeRegressor(max_depth=self.max_depth, 
                                         **self.trees_params)
            tree.fit(X[:, ind], targ)
            pred = tree.predict(X[:, ind])
            c_min = minimize_scalar(lambda c: np.sum((curr_func + c * pred - y) ** 2)).x
            self.trees_list.append(tree)
            self.coeff_list.append(self.learning_rate * c_min)
            curr_func += self.learning_rate * c_min * tree.predict(X[:, ind])
            
    def predict(self, X, y_val):
        """
        X : numpy ndarray
            Array of size n_objects, n_features
            
        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        self.acc_list = []
        pred_list = []
        num = 0
        for tree in self.trees_list:
            pred_list.append(tree.predict(X[:, self.ind_list[num]]))
            num += 1
            y_pred = np.sum(np.array(self.coeff_list)[:num, np.newaxis] * np.array(pred_list), axis=0)
            self.acc_list.append(np.sqrt(mean_squared_error(y_val, y_pred)))
        return np.sum(np.array(self.coeff_list)[:, np.newaxis] * np.array(pred_list), axis=0)
