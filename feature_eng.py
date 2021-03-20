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

input_path = '../data/' # original dataset
output_path = '../mod_data/' # dataset after feature engineering
result_path = '../results/'

train = pd.read_csv(input_path+'train.csv')
test = pd.read_csv(input_path+'test.csv')


print('training dataset size {}, {}'.format(train.shape[0],train.shape[1]))
print('test datase size {}, {}'.format(test.shape[0], test.shape[1]))
print('train columns\n', train.columns)
print('test columns\n', test.columns)
print('columns types\n', train.dtypes)
print('train.count\n', train.count())


max_dur = train['trip_duration'].max() / 3600
min_dur = train['trip_duration'].min() / 3600
train['log_train_trip_duration'] = np.log(train['trip_duration'].values + 1)

plt.figure(figsize=(5,5))
plt.hist(train['log_train_trip_duration'].values, bins = 500)
plt.xlabel('log trip duration', fontsize=12)
plt.ylabel('count', fontsize=12)
plt.title('Hist of trip duration', fontsize=12)
plt.tight_layout()
# plt.savefig(result_path+'hist_of_trip_duration'+'.png', dpi=100)
plt.show()



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


test['pickup_datetime'] = pd.to_datetime(test.pickup_datetime)
test.loc[:,'pickup_date'] = test['pickup_datetime'].dt.date
test.loc[:,'pickup_year'] = test['pickup_datetime'].dt.year
test.loc[:,'pickup_month'] = test['pickup_datetime'].dt.month
test.loc[:,'pickup_day'] = test['pickup_datetime'].dt.day
test.loc[:, 'pickup_weekday'] = test['pickup_datetime'].dt.weekday
test.loc[:, 'pickup_weekofyear'] = test['pickup_datetime'].dt.isocalendar().week
test.loc[:, 'pickup_hour'] = test['pickup_datetime'].dt.hour
test.loc[:, 'pickup_minute'] = test['pickup_datetime'].dt.minute
test.loc[:, 'pickup_dt'] = (test['pickup_datetime']-test['pickup_datetime'].min()).dt.total_seconds()# what is this?
test.loc[:, 'pickup_week_hour'] = test['pickup_weekday']*24+test['pickup_hour']
 # what is this?
test['store_and_fwd_flag'] = 1*(test.store_and_fwd_flag.values=='Y')

# PCA on longitude and latitude coordinates
coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values, train[['dropoff_latitude', 'dropoff_longitude']].values, test[['pickup_latitude', 'pickup_longitude']].values, test[['pickup_latitude', 'pickup_longitude']].values))
pca = PCA().fit(coords)
train['pickup_pca0'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:,0]
train['pickup_pca1'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:,1]
train['dropoff_pca0'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:,0]
train['dropoff_pca1'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:,1]

test['pickup_pca0'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 0]
test['pickup_pca1'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 1]
test['dropoff_pca0'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
test['dropoff_pca1'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 1]

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
train.loc[:, 'pca_manhattan'] = np.abs(train['dropoff_pca1'] - train['pickup_pca1']) + np.abs(train['dropoff_pca0'] - train['pickup_pca0'])
train.loc[:, 'center_latitude'] = (train['pickup_latitude'].values + train['dropoff_latitude'].values) / 2
train.loc[:, 'center_longitude'] = (train['pickup_longitude'].values + train['dropoff_longitude'].values) / 2

test.loc[:, 'distance_haversine'] = haversine_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)
test.loc[:, 'distance_dummy_manhattan'] = dummy_manhattan_distance(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)
test.loc[:, 'direction'] = bearing_array(test['pickup_latitude'].values, test['pickup_longitude'].values, test['dropoff_latitude'].values, test['dropoff_longitude'].values)
test.loc[:, 'pca_manhattan'] = np.abs(test['dropoff_pca1'] - test['pickup_pca1']) + np.abs(test['dropoff_pca0'] - test['pickup_pca0'])
test.loc[:, 'center_latitude'] = (test['pickup_latitude'].values + test['dropoff_latitude'].values) / 2
test.loc[:, 'center_longitude'] = (test['pickup_longitude'].values + test['dropoff_longitude'].values) / 2

# cluster pickup and dropoff positions.
sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])

train.loc[:, 'pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])
train.loc[:, 'dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])
test.loc[:, 'pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])
test.loc[:, 'dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])


city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)
N = 10000

plt.figure(figsize=(5,5))
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.scatter(train.pickup_longitude.values[:N], train.pickup_latitude.values[:N], s=10, lw=0,
           c=train.pickup_cluster[:N].values, cmap='tab20', alpha=0.2)
ax.set_xlim(city_long_border)
ax.set_ylim(city_lat_border)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.savefig(result_path+'map_of_location'+'.png', dpi=100)
plt.show()

# Speed 
'''
speed can be calauted with train set. However, it is not available in test set. 
What we can do: Find averge speed w.r.t location, time, season, and used as feature in train and test set.
'''

train.loc[:, 'avg_speed_h'] = 1000 * train['distance_haversine'] / train['trip_duration']
train.loc[:, 'avg_speed_m'] = 1000 * train['distance_dummy_manhattan'] / train['trip_duration']
train.loc[:, 'pickup_lat_bin'] = np.round(train['pickup_latitude'], 2)
train.loc[:, 'pickup_long_bin'] = np.round(train['pickup_longitude'], 2)
train.loc[:, 'center_lat_bin'] = np.round(train['center_latitude'], 2)
train.loc[:, 'center_long_bin'] = np.round(train['center_longitude'], 2)
train.loc[:, 'pickup_dt_bin'] = (train['pickup_dt'] // (3 * 3600))


test.loc[:, 'pickup_lat_bin'] = np.round(test['pickup_latitude'], 2)
test.loc[:, 'pickup_long_bin'] = np.round(test['pickup_longitude'], 2)
test.loc[:, 'center_lat_bin'] = np.round(test['center_latitude'], 2)
test.loc[:, 'center_long_bin'] = np.round(test['center_longitude'], 2)
test.loc[:, 'pickup_dt_bin'] = (test['pickup_dt'] // (3 * 3600))
# Whether the dropoff position is ariport. Add boolean feature to indicate airport.

feature_names = list(train.columns)
print(np.setdiff1d(train.columns, test.columns))

do_not_use_for_training = ['id', 'log_trip_duration', 'pickup_datetime', 'dropoff_datetime', 'trip_duration', 'check_trip_duration', 'pickup_date', 'avg_speed_h', 'avg_speed_m', 'pickup_lat_bin', 'pickup_long_bin', 'center_lat_bin', 'center_long_bin','pickup_dt_bin', 'pickup_datetime_group']

feature_names = [f for f in train.columns if f not in do_not_use_for_training]
# only keep nuerial features and features shared by train and test set.
# train.to_csv(output_path+'train.csv', index = False)
# test.to_csv(output_path+'test.csv', index = False)

# cols = feature_names + ['log_train_trip_duration']
# cols = ['passenger_count', 'pickup_cluster', 'dropoff_cluster', 'pickup_date', 'pickup_hour', 'log_train_trip_duration']
cols = ['pickup_longitude', 'pickup_latitude','dropoff_longitude','dropoff_latitude', 'pickup_hour', 'distance_dummy_manhattan', 'pickup_cluster', 'dropoff_cluster','log_train_trip_duration']
cols_1 = ['pickup_longitude', 'pickup_latitude','dropoff_longitude','dropoff_latitude', 'log_train_trip_duration'] 
cols_2 = ['pickup_hour', 'distance_dummy_manhattan', 'pickup_cluster', 'dropoff_cluster','log_train_trip_duration']

# sns.pairplot(train[cols_1].iloc[:1000,:], height = 1)
# plt.show()

# sns.pairplot(train[cols_2].iloc[:1000,:], height = 1)
# plt.show()



# cm = np.corrcoef(train[cols].iloc[:200,:].values.T)
# sns.set(font_scale=1.5)
# hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels = cols, xticklabels=cols)
# plt.show()



# def calc_vif(X):
#     # Calculating VIF
#     vif = pd.DataFrame()
#     vif["variables"] = X.columns
#     vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
#     return(vif)

# X = train[cols].iloc[:N, :-1]
# VIF = calc_vif(X)

cols = ['pickup_longitude', 'pickup_latitude','dropoff_longitude','dropoff_latitude', 'pickup_hour', 'distance_dummy_manhattan', 'pickup_cluster', 'dropoff_cluster']

#### important score from random tree
x_train = train[cols].iloc[:N, :-1]
y_train = train['log_train_trip_duration'].iloc[:N].values

feat_labels = cols[:-1]
forest = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
forest.fit(x_train, y_train)
imp_scores = forest.feature_importances_ 
indices = np.argsort(imp_scores)[::-1]
for f in range(x_train.shape[1]):
    print('%2d) %-*s %f'%(f+1, 30, feat_labels[f], imp_scores[indices[f]]))


# plt.figure(figsize=(5,5))
# plt.title('Feature Importance')
# plt.bar(range(x_train.shape[1]), imp_scores[indices], color='lightblue', align = 'center')
# plt.xticks(range(x_train.shape[1]), feat_labels, rotation=90)
# plt.xlim([-1, x_train.shape[1]])
# plt.tight_layout()
# plt.show()


# PCA


x, y = train[cols].iloc[:N, :-1].values, train['log_train_trip_duration'].iloc[:N].values

# x_features = feature_names.copy()
# x_features.remove('log_train_trip_duration')
# N = 10000
# train_x = train[x_features].iloc[:N,:-1]
# train_y = train['log_train_trip_duration'].iloc[:N]
# train_x.to_csv(output_path+'train_x.csv', index=False)
# train_y.to_csv(output_path+'train_y.csv', index=False)

x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.4, random_state=0)
sc = StandardScaler()
x_tr_std = sc.fit_transform(x_tr)
x_te_std = sc.fit_transform(x_te)

cov_mat = np.cov(x_tr_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
tot = sum(eig_vals)
var_exp = [(i/tot) for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# plt.figure(figsize=(5,5))
# plt.bar(range(1, x.shape[1]+1), var_exp, alpha=0.5, align='center', label='individual explained variance')
# plt.step(range(1,x.shape[1]+1), cum_var_exp, where='mid', label='cumulative explained variance')
# plt.ylabel('Explained variance ratio')
# plt.xlabel('Principal componnets')
# plt.legend(loc='best')
# plt.show()
### modelling




