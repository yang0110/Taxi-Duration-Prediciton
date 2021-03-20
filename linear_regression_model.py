import numpy as np
import pandas as pd 
import numpy as np 
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', context='notebook')
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import xgboost as xgb
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

input_path = '../mod_data/'
x = pd.read_csv(input_path+'train_x.csv')
y = pd.read_csv(input_path+'train_y.csv')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
lr_model = LinearRegression()

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

kfold = 10
cv_results = cross_val_score(lr_model, x_train, y_train, cv=kfold, scoring='neg_mean_squared_error')
print('cv_mean {}, cv std {}'.format(cv_results.mean(), cv_results.std()))



# class LinearRegressionGD(object):
#     def __init__(self, eta=0.001, n_iter=20):

#         self.eta = eta
#         self.n_iter = n_iter 

#     def fit(self, x, y):
        
#         self.w_ = np.zeros(1+x.shape[1])
#         self.cost_ = []
#         for i in range(self.n_iter):
#             output = self.net_input(x)
#             errors = (y-output)
#             self.w_[1:] += self.eta*x.T.dot(errors)
#             self.w_[0] += self.eta*errors.sum()
#             cost = (errors**2).sum()/2.0
#             self.cost_.append(cost)

#     def net_input(self, x):

#         return np.dot(x, self.w_[1:]+self.w_[0])

#     def predict(self, x):

#         return self.net_input(x)


# lr = LinearRegressionGD()
# lr.fit(x_tr_std, y_tr)
# plt.figure(figsize=(5,5))
# plt.plot(range(1, lr.n_iter+1), lr.cost_)
# plt.ylabel('SSE', fontsize=12)
# plt.xlabel('Epoch', fontsize=12)
# plt.show()