import numpy as np 
import pandas as pd 
import seaborn as sns 
from tqdm.notebook import tqdm 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
result_path = '../results/'
input_path = '../mod_data/'
x = pd.read_csv(input_path+'train_x.csv')
y = pd.read_csv(input_path+'train_y.csv')
train = x.copy()
train['log_train_trip_duration'] = y['log_train_trip_duration'] 

cols = train.columns
cols_1 = ['vendor_id', 'passenger_count', 'pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude', 'store_and_fwd_flag']
cols_2 = ['pickup_month', 'pickup_day', 'pickup_weekday',
       'pickup_weekofyear', 'pickup_hour', 'pickup_minute']
cols_3 = ['pickup_dt',
       'pickup_week_hour', 'pickup_pca0', 'pickup_pca1', 'dropoff_pca0',
       'dropoff_pca1', 'distance_haversine']

cols_4 = ['distance_dummy_manhattan',
       'direction', 'pca_manhattan', 'center_latitude', 'center_longitude',
       'pickup_cluster', 'log_train_trip_duration']

sub_cols = ['pickup_longitude', 'pickup_latitude',
       'dropoff_longitude', 'dropoff_latitude','distance_haversine','distance_dummy_manhattan',
       'direction','pickup_cluster','log_train_trip_duration']
sns.pairplot(train[cols_1].iloc[:1000,:], height = 1)
plt.show()

sns.pairplot(train[cols_2].iloc[:1000,:], height = 1)
plt.show()

sns.pairplot(train[cols_3].iloc[:1000,:], height = 1)
plt.show()

sns.pairplot(train[cols_4].iloc[:1000,:], height = 1)
plt.show()


plt.figure(figsize=(5,5))
cm = np.corrcoef(train[cols_1].iloc[:1000,:].values.T)
sns.set(font_scale=0.7)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels = cols_1, xticklabels=cols_1)
plt.tight_layout()
plt.savefig(result_path+'corr1'+'.png', dpi=100)
plt.show()

plt.figure(figsize=(5,5))
cm = np.corrcoef(train[cols_2].iloc[:1000,:].values.T)
sns.set(font_scale=0.7)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels = cols_2, xticklabels=cols_2)
plt.tight_layout()
plt.savefig(result_path+'corr2'+'.png', dpi=100)
plt.show()

plt.figure(figsize=(5,5))
cm = np.corrcoef(train[cols_3].iloc[:1000,:].values.T)
sns.set(font_scale=0.7)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels = cols_3, xticklabels=cols_3)
plt.tight_layout()
plt.savefig(result_path+'corr3'+'.png', dpi=100)
plt.show()

plt.figure(figsize=(5,5))
cm = np.corrcoef(train[cols_4].iloc[:1000,:].values.T)
sns.set(font_scale=0.7)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels = cols_4, xticklabels=cols_4)
plt.tight_layout()
plt.savefig(result_path+'corr4'+'.png', dpi=100)
plt.show()

# def calc_vif(X):
#     # Calculating VIF
#     vif = pd.DataFrame()
#     vif["variables"] = X.columns
#     vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
#     return(vif)

# X = train[sub_cols]
# VIF = calc_vif(X)


#### important score from random tree


feat_labels = sub_cols[:-1]
forest = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
forest.fit(x[feat_labels], y)
imp_scores = forest.feature_importances_ 
indices = np.argsort(imp_scores)[::-1]
for f in range(x[feat_labels].shape[1]):
    print('%2d) %-*s %f'%(f+1, 30, feat_labels[f], imp_scores[indices[f]]))


plt.figure(figsize=(5,5))
plt.title('Feature Importance')
plt.bar(range(x[feat_labels].shape[1]), imp_scores[indices], color='lightblue', align = 'center')
plt.xticks(range(x[feat_labels].shape[1]), feat_labels, rotation=90)
plt.xlim([-1, x[feat_labels].shape[1]])
plt.tight_layout()
plt.show()



x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.4, random_state=0)
sc = StandardScaler()

x_tr_std = sc.fit_transform(x_tr)
x_te_std = sc.fit_transform(x_te)

cov_mat = np.cov(x_tr_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
tot = sum(eig_vals)
var_exp = [(i/tot) for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

plt.figure(figsize=(5,5))
plt.bar(range(1, x.shape[1]+1), var_exp, alpha=0.5, align='center', label='individual explained variance')
plt.step(range(1,x.shape[1]+1), cum_var_exp, where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal componnets')
plt.legend(loc='best')
plt.show()
