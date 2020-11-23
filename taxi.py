import pandas as pd 
import numpy as np 
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white')
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error, mean_squared_error
from sklearn.cluster import MiniBatchKMeans
input_path = '../data/' # original dataset
output_path = '../mod_data/' # dataset after feature engineering
result_path = '../results/'

train = pd.read_csv(input_path+'train.csv')

# train.head(2)

# print('training dataset size {}, {}'.format(train.shape[0],train.shape[1]))
# print('train columns\n', train.columns)
# print('columns types\n', train.dtypes)
# print('train.count\n', train.count())

max_dur = train['trip_duration'].max() / 3600
min_dur = train['trip_duration'].min() / 3600
train['log_trip_duration'] = np.log(train['trip_duration'].values + 1)

# plt.figure(figsize=(6, 4))
# plt.hist(train['trip_duration'].values, bins = 100)
# plt.xlabel('Trip duration', fontsize=12)
# plt.ylabel('Count', fontsize=12)
# plt.title('Train', fontsize=12)
# plt.tight_layout()
# plt.savefig(result_path+'train_trip_duration'+'.png', dpi=100)
# plt.show()


# plt.figure(figsize=(6, 4))
# plt.hist(train['log_trip_duration'].values, bins = 100)
# plt.xlabel('log(trip duration)', fontsize=12)
# plt.ylabel('Count', fontsize=12)
# plt.title('Train', fontsize=12)
# plt.tight_layout()
# plt.savefig(result_path+'train_log_duration'+'.png', dpi=100)
# plt.show()

# feature reformatting
train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
train.loc[:,'pickup_date'] = train['pickup_datetime'].dt.date
train.loc[:,'pickup_year'] = train['pickup_datetime'].dt.year
train.loc[:,'pickup_month'] = train['pickup_datetime'].dt.month
train.loc[:,'pickup_day'] = train['pickup_datetime'].dt.day
train.loc[:, 'pickup_weekday'] = train['pickup_datetime'].dt.weekday
train.loc[:, 'pickup_weekofyear'] = train['pickup_datetime'].dt.isocalendar().week
train.loc[:, 'pickup_hour'] = train['pickup_datetime'].dt.hour
train.loc[:, 'pickup_minute'] = train['pickup_datetime'].dt.minute
train.loc[:, 'pickup_dt'] = (train['pickup_datetime']-train['pickup_datetime'].min()).dt.total_seconds()# what is this?
train.loc[:, 'pickup_week_hour'] = train['pickup_weekday']*24+train['pickup_hour']
 # what is this?
train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)
 # not available in test set, so not used in feature enigeer and train model. But, can be used in exploratory data analysis.
train['store_and_fwd_flag'] = 1*(train.store_and_fwd_flag.values=='Y')

used_columns = ['vendor_id', 
       'passenger_count', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag',
		'pickup_year',
       'pickup_month', 'pickup_day', 'pickup_weekday', 'pickup_weekofyear',
       'pickup_hour', 'pickup_minute', 'pickup_dt', 'pickup_week_hour', 'log_trip_duration',]

# mod_train = train[used_columns]
# x = mod_train.iloc[:, :-1]
# y = mod_train.iloc[:, -1]

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

# lr_model = LinearRegression()
# lr_model.fit(x_train, y_train)

# y_hat = lr_model.predict(x_train)
# error = mean_squared_error(y_hat, y_train)
# error = np.sqrt(error)
# print('RMSLE:', error)

# y_hat = lr_model.predict(x_test)
# error = mean_squared_error(y_hat, y_test)
# error = np.sqrt(error)
# print('RMSLE:', error)


# Distance between pickup and dropoff
def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b

def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

train.loc[:, 'distance_haversine'] = haversine_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
train.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
train.loc[:, 'direction'] = bearing_array(train['pickup_latitude'].values, train['pickup_longitude'].values, train['dropoff_latitude'].values, train['dropoff_longitude'].values)
train.loc[:, 'center_latitude'] = (train['pickup_latitude'].values + train['dropoff_latitude'].values) / 2
train.loc[:, 'center_longitude'] = (train['pickup_longitude'].values + train['dropoff_longitude'].values) / 2


# find clusters of pick up and drop off locations
coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values, train[['dropoff_latitude', 'dropoff_longitude']].values))

sample_ind = np.random.permutation(len(coords))[:500000]

kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])

train.loc[:, 'pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])
train.loc[:, 'dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])


city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)
N = 10000

# plt.figure(figsize=(6, 4))
# fig, ax = plt.subplots(ncols=1, nrows=1)
# ax.scatter(train.pickup_longitude.values[:N], train.pickup_latitude.values[:N], s=10, lw=0, c=train.pickup_cluster[:N].values, cmap='tab20', alpha=0.2)
# ax.set_xlim(city_long_border)
# ax.set_ylim(city_lat_border)
# ax.set_xlabel('Longitude', fontsize=12)
# ax.set_ylabel('Latitude', fontsize=12)
# plt.tight_layout()
# plt.savefig(result_path+'map_of_location'+'.png', dpi=100)
# plt.show()

train['pick_airport'] = 0 
train.loc[(train.pickup_longitude.values>-73.8) & (train.pickup_latitude<40.67), 'pick_airport'] = 1

train['dropoff_airport'] = 0 
train.loc[(train.dropoff_longitude.values>-73.8) & (train.dropoff_latitude<40.67), 'dropoff_airport'] = 1

used_columns = ['vendor_id', 
       'passenger_count', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag',
      'pickup_year', 'pickup_month',
       'pickup_day', 'pickup_weekday', 'pickup_weekofyear', 'pickup_hour',
       'pickup_minute', 'pickup_dt', 'pickup_week_hour', 
       'distance_haversine', 'distance_dummy_manhattan', 'direction',
       'center_latitude', 'center_longitude', 'pickup_cluster',
       'dropoff_cluster', 'pick_airport', 'dropoff_airport', 'log_trip_duration']

# mod_train = train[used_columns]

# x = mod_train.iloc[:, :-1]
# y = mod_train.iloc[:, -1]

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

# lr_model = LinearRegression()
# lr_model.fit(x_train, y_train)

# y_hat = lr_model.predict(x_train)
# error = mean_squared_error(y_hat, y_train)
# error = np.sqrt(error)
# print('RMSLE:', error)

# y_hat = lr_model.predict(x_test)
# error = mean_squared_error(y_hat, y_test)
# error = np.sqrt(error)
# print('RMSLE:', error)


# col1 = ['pickup_hour',
#         'distance_dummy_manhattan', 'direction', 'log_trip_duration']
# col2 = ['pickup_cluster',
#         'dropoff_cluster', 'pick_airport', 'dropoff_airport', 'log_trip_duration']
# sns.pairplot(mod_train[col1].iloc[:1000,:], height = 1)
# plt.show()

# sns.pairplot(mod_train[col2].iloc[:1000,:], height = 1)
# plt.show()


# short_col1 = ['hour',
#         'manhattan', 'direction', 'duration']
# short_col2 = ['pickcluster',
#         'dropcluster', 'pickairport', 'drop_airport', 'duration']

# plt.figure(figsize=(5,5))
# cm = np.corrcoef(train[col1].iloc[:1000,:].values.T)
# sns.set(font_scale=0.7)
# hm = sns.heatmap(cm, cbar=False, annot=True, square=True, fmt='.2f', annot_kws={'size': 12}, yticklabels = short_col1, xticklabels=short_col1)
# plt.tight_layout()
# plt.savefig(result_path+'corr1'+'.png', dpi=100)
# plt.show()

# plt.figure(figsize=(5,5))
# cm = np.corrcoef(train[col2].iloc[:1000,:].values.T)
# sns.set(font_scale=0.7)
# hm = sns.heatmap(cm, cbar=False, annot=True, square=True, fmt='.2f', annot_kws={'size': 12}, yticklabels = short_col2, xticklabels=short_col2)
# plt.tight_layout()
# plt.savefig(result_path+'corr2'+'.png', dpi=100)
# plt.show()

# mod_train = train[used_columns]

# x = mod_train.iloc[:, :-1]
# y = mod_train.iloc[:, -1]

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)

# from sklearn.linear_model import Lasso 
# lasso = Lasso(alpha=0.01)
# lasso.fit(x_train, y_train)
# y_hat = lasso.predict(x_train)
# error = mean_squared_error(y_hat, y_train)
# print('error', np.sqrt(error))
# y_hat = lasso.predict(x_test)
# error = mean_squared_error(y_hat, y_test)
# print('error', np.sqrt(error))

# coefs = lasso.coef_
# indices = np.argsort(np.abs(coefs))[::-1]
# plt.figure(figsize=(6,4))
# plt.bar(range(x_train.shape[1]), coefs[indices], color='lightblue', align='center')
# plt.xticks(range(x_train.shape[1]), used_columns, rotation=90)
# plt.xlabel('feature index', fontsize=12)
# plt.ylabel('coefficient', fontsize=12)
# plt.tight_layout()
# plt.savefig(result_path+'lasso_coef'+'.png', dpi=100)
# plt.show()


# from sklearn.ensemble import RandomForestRegressor
# from sklearn.utils.random import sample_without_replacement
# rf = RandomForestRegressor()
# sub_index = sample_without_replacement(x_train.shape[1], int(x_train.shape[1]/10))
# sub_x_train = x_train[sub_index]
# sub_y_train = y_train[sub_index]
# rf.fit(x_train, y_train)
# imp = rf.feature_importances_
# indices = np.argsort(imp)[::-1]

# plt.figure(figsize=(6,4))
# plt.bar(range(x_train.shape[1]), imp[indices], color='lightblue', align='center')
# plt.xticks(range(x_train.shape[1]), used_columns, rotation=90)
# plt.xlim([-1, x_train.shape[1]])
# plt.xlabel('feature index', fontsize=12)
# plt.ylabel('feature importance', fontsize=12)
# plt.tight_layout()
# plt.savefig(result_path+'random_forest_feature_importance'+'.png', dpi=100)
# plt.show()


# cov_mat = np.cov(x_train.T)
# eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
# tot = sum(eigen_vals)
# var_exp = [(i/tot) for i in sorted(eigen_vals, reverse=True)]
# cum_var_exp = np.cumsum(var_exp)

# plt.figure(figsize=(6,4))
# plt.bar(range(x_train.shape[1]), var_exp, alpha=0.5, color='lightblue', align='center', label='individual explained variance')
# plt.step(range(x_train.shape[1]), cum_var_exp, where='mid', color='lightblue', label='cumulative explained variance')
# plt.ylabel('Explained variance ratio', fontsize=12)
# plt.xlabel('Principle Components', fontsize=12)
# plt.legend(loc=0, fontsize=12)
# plt.tight_layout()
# plt.savefig(result_path+'pca_score'+'.png', dpi=100)
# plt.show()

# from utils import SBS
# lr = LinearRegression()
# sbs = SBS(lr, k_features=1)
# sbs.fit(sub_x_train, sub_y_train)

# k_feat = [len(k) for k in sbs.subsets_]
# plt.figure(fogsize=(6,4))
# plt.plot(k_feat, sbs_scores_, marker='o')
# plt.ylabel('MSE', fontsize=12)
# plt.xlabel('Number of features', fontsize=12)
# plt.tight_layout()
# plt.show()


from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures 
N=10000
mod_train = train[used_columns]
x = mod_train.iloc[:N, :-1]
y = mod_train.iloc[:N, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

lr = linear_model.LinearRegression()
lr.fit(x_train, y_train)
lr_y = lr.predict(x_test)
lr_yt = lr.predict(x_train)

ridge = linear_model.Ridge(alpha=0.1)
ridge.fit(x_train, y_train)
ridge_y = ridge.predict(x_test)
ridge_yt = ridge.predict(x_train)

lasso = linear_model.Lasso(alpha=0.1)
lasso.fit(x_train, y_train)
lasso_y = lasso.predict(x_test)
lasso_yt = lasso.predict(x_train)

poly = PolynomialFeatures(degree=2)
poly_train = poly.fit_transform(x_train)
poly_test = poly.transform(x_test)
lr = linear_model.LinearRegression()
lr.fit(poly_train, y_train)
poly_y = lr.predict(poly_test)
poly_yt = lr.predict(poly_train)

from sklearn import svm 
svmr = svm.SVR()
svmr.fit(x_train, y_train)
svmr_y = svmr.predict(x_test)
svmr_yt = svmr.predict(x_train)

svmr2 = svm.SVR(kernel='rbf')
svmr2.fit(x_train, y_train)
svmr2_y = svmr2.predict(x_test)
svmr2_yt = svmr2.predict(x_train)

from sklearn import neighbors 
knn = neighbors.KNeighborsRegressor(n_neighbors=10)
knn.fit(x_train, y_train)
knn_y = knn.predict(x_test)
knn_yt = knn.predict(x_train)

from sklearn import tree 
tree_model = tree.DecisionTreeRegressor(max_depth=4)
tree_model.fit(x_train, y_train)
tree_y = tree_model.predict(x_test)
tree_yt = tree_model.predict(x_train)

from sklearn.ensemble import AdaBoostRegressor
ada = AdaBoostRegressor(tree.DecisionTreeRegressor(max_depth=4), n_estimators=10)
ada.fit(x_train, y_train)
ada_y = ada.predict(x_test)
ada_yt = ada.predict(x_train)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(max_depth=5)
rf.fit(x_train, y_train)
rf_y = rf.predict(x_test)
rf_yt = rf.predict(x_train)

yt_list = [lr_yt, ridge_yt, lasso_yt, poly_yt, svmr_yt, svmr2_yt, tree_yt, knn_yt, ada_yt, rf_yt]
train_error_list = [np.sqrt(mean_squared_error(y, y_train)) for y in yt_list]

y_list = [lr_y, ridge_y, lasso_y, poly_y, svmr_y, svmr2_y, tree_y, knn_y, ada_y, rf_y]
test_error_list = [np.sqrt(mean_squared_error(y, y_test)) for y in y_list]

# plt.figure(figsize=(6,4))
# plt.bar(np.arange(len(model_list)), test_error_list, color='lightblue', align='center')
# plt.xticks(np.arange(len(model_list)), model_list, rotation=45)
# # plt.xlim([-1, x_train.shape[1]])
# plt.xlabel('Models', fontsize=12)
# plt.ylabel('Test error', fontsize=12)
# plt.tight_layout()
# # plt.savefig(result_path+'models_test_error'+'.png', dpi=100)
# plt.show()

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim 
class Dataset_py(Dataset):
  def __init__(self, x, y):
    self.x = torch.tensor(x, dtype=torch.float32)
    self.y = torch.tensor(y, dtype=torch.float32)

  def __len__(self):
    return len(self.x)

  def __getitem__(self, idx):
    return self.x[idx], self.y[idx]



y_train_array, y_test_array = y_train.values, y_test.values
y_train_array = y_train_array.reshape((len(y_train_array), 1))
y_test_array = y_test_array.reshape((len(y_test_array), 1))

batch_size = 32
train_ds = Dataset_py(x_train, y_train_array)
# test_ds = Dataset_py(x_test, y_test_array)
train_dl = DataLoader(train_ds, batch_size=batch_size)
xtrain_t = torch.tensor(x_train, dtype=torch.float32)
xtest_t = torch.tensor(x_test, dtype=torch.float32)

class MLP(nn.Module):
  def __init__(self, input_size, output_size):
    super(MLP, self).__init__()
    self.net = nn.Sequential(
    nn.Linear(input_size, 64), 
    nn.ReLU(), 
    nn.Linear(64, 32), 
    nn.ReLU(),
    nn.Linear(32, output_size),
    )

  def forward(self, x):
    pred = self.net(x)
    return pred 


nn_model = MLP(x_train.shape[1], 1)
cost = nn.MSELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.001)
loss_list = []

epoch_num = 300
for epoch in range(epoch_num):
  for x_batch, y_batch in train_dl:
      pred = nn_model(x_batch)
      loss = torch.sqrt(cost(pred, y_batch))
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
  print('epoch %s, error %s'%(epoch, loss.item()))
  loss_list.append(loss.item())

nn_model.eval()
nn_y = nn_model(xtest_t).detach().numpy()
nn_yt = nn_model(xtrain_t).detach().numpy()
nn_test_error = np.sqrt(mean_squared_error(nn_y, y_test_array))
nn_train_error = np.sqrt(mean_squared_error(nn_yt, y_train_array))
print('train_error, test_error', nn_train_error, nn_test_error)
train_error_list.append(nn_train_error)
test_error_list.append(nn_test_error)
#0.39, 0.44
plt.figure(figsize=(6,4))
plt.plot(loss_list)
plt.xlabel('Epcoh num', fontsize=12)
plt.ylabel('Loss',fontsize=12)
plt.tight_layout()
plt.savefig(result_path+'nn_learning_curve'+'.png', dpi=100)
plt.show()

import xgboost as xgb
# N=50000
# mod_train = train[used_columns]
x = mod_train.iloc[:N, :-1].values
y = mod_train.iloc[:N, -1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_val = sc.transform(x_val)
x_test = sc.transform(x_test)

dtrain = xgb.DMatrix(x_train, label=y_train)
dvalid = xgb.DMatrix(x_val, label=y_val)
dtest = xgb.DMatrix(x_test, label=y_test)

watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
xgb_pars = {'min_child_weight': 10, 'eta': 0.01, 'colsample_bytree': 0.3, 'max_depth': 6, 'subsample': 0.5, 'lambda': 1, 'booster': 'gbtree', 'eval_metric': 'rmse', 'objective':'reg:squarederror'}

model = xgb.train(xgb_pars, dtrain, 2000, watchlist, early_stopping_rounds=50, maximize=False, verbose_eval=10)
print('Modeling RMSLE %.5f' % model.best_score)
xgb_train_error = model.best_score
xgb_y = model.predict(dtest)
xgb_test_error = np.sqrt(mean_squared_error(xgb_y, y_test))
print('train error %s, test_error %s'%(xgb_train_error, xgb_test_error))
train_error_list.append(xgb_train_error)
test_error_list.append(xgb_test_error)
# 0.37, 0.39


import lightgbm as lgb
d_train = lgb.Dataset(x_train, y_train)
lgb_params = {
              'learning_rate': 0.1, # try 0.2
              'max_depth': 5,
              'num_leaves': 10, 
              'objective': 'regression',
              'metric': {'rmse'},
              'feature_fraction': 0.5,
              'bagging_fraction': 0.5,
              #'bagging_freq': 5,
              'max_bin': 100}       # 1000
n_rounds = 100
model_lgb = lgb.train(lgb_params, 
                      d_train, 
                      # feval=lgb_rmsle_score, 
                      num_boost_round=n_rounds)

light_yt = model_lgb.predict(x_train)
light_y = model_lgb.predict(x_test)
light_train_error = np.sqrt(mean_squared_error(light_yt, y_train))
light_test_error = np.sqrt(mean_squared_error(light_y, y_test))
print('train error %s, test error %s'%(light_train_error, light_test_error))
train_error_list.append(light_train_error)
test_error_list.append(light_test_error)

model_list = ['lr', 'ridge', 'lasso', 'poly_lr', 'svm', 'svm_rbf', 'decision tree', 'knn', 'adaboost', 'random forest', 'nn', 'xgboost', 'lightbgm']
print('models train error', train_error_list)
print('models test error', test_error_list)
plt.figure(figsize=(6,4))
plt.bar(np.arange(len(model_list)), train_error_list, width=0.2,  color='lightblue', align='center', label='Train')
plt.bar(np.arange(len(model_list))-0.2, test_error_list, width=0.2, color='y', align='center', label='Test')
plt.xticks(np.arange(len(model_list)), model_list, rotation=90)
# plt.xlim([-1, x_train.shape[1]])
plt.xlabel('Models', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.legend(loc=0, fontsize=12)
plt.tight_layout()
plt.savefig(result_path+'models_error'+'.png', dpi=100)
plt.show()

#models train error [0.6013599518650662, 0.6014082819055208, 0.6475190748543749, 0.5092668066488578, 0.4457634185693864, 0.4457634185693864, 0.49651048477502346, 0.5616639000820584, 0.4966186291973662, 0.45684375286468004, 0.38533719314699144, 0.481919, 0.384659218368034]
#models test error [0.5591325876082729, 0.5591592374453985, 0.6015741536508901, 0.7416791749281934, 0.4584919641417047, 0.4584919641417047, 0.5052637468808265, 0.6056340034203053, 0.5209706906928969, 0.4771346913889011, 0.4852695048116297, 0.4176672330467144, 0.4240023085171282]