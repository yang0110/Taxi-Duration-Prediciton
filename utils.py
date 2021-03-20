import numpy as np

class LinearRegressionGD(object):
    def __init__(self, eta=0.001, n_iter=20):

        self.eta = eta
        self.n_iter = n_iter 

    def fit(self, x, y):
        
        self.w_ = np.zeros(1+x.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(x)
            errors = (y-output)
            self.w_[1:] += self.eta*x.T.dot(errors)
            self.w_[0] += self.eta*errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)

    def net_input(self, x):

        return np.dot(x, self.w_[1:]+self.w_[0])

    def predict(self, x):

        return self.net_input(x)


from sklearn.base import clone
from itertools import combinations 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

class SBS():
    def __init__(self, estimator, k_features, scoring=mean_squared_error, test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, x, y):
        x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=self.test_size, random_state= self.random_state)
        dim = x_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets = [self.indices_]
        score = self._calc_score(x_train, y_train, x_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []
            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(x_train, y_train, x_test, y_test, p)
                scores.append(score)
                subsets.append(p)

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        return self 

    def transform(self, x):
        return x[:, self.indices_]

    def _calc_score(self, x_train, y_train, x_test, y_test, indices):
        self.estimator.fit(x_train[:, indices], y_train)
        y_pred = self.estimator.predict(x_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return -score




















