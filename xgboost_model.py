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
result_path = '../results/'
input_path = '../mod_data/'
x = pd.read_csv(input_path+'train_x.csv')
y = pd.read_csv(input_path+'train_y.csv')

feature_names = x.columns
Xtr, Xv, ytr, yv = train_test_split(x.values, y.values, test_size=0.2, random_state=1987)
dtrain = xgb.DMatrix(Xtr, label=ytr)
dvalid = xgb.DMatrix(Xv, label=yv)
# dtest = xgb.DMatrix(test[feature_names].values)
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

xgb_pars = {'min_child_weight': 50, 'eta': 0.3, 'colsample_bytree': 0.3, 'max_depth': 10,
            'subsample': 0.8, 'lambda': 1., 'nthread': 4, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'rmse', 'objective': 'reg:linear'}

t0 = dt.datetime.now()
model = xgb.train(xgb_pars, dtrain, 60, watchlist, early_stopping_rounds=50,
                  maximize=False, verbose_eval=10)

print('Modeling RMSLE %.5f' % model.best_score)
t1 = dt.datetime.now()
print('Training time: %i seconds' % (t1 - t0).seconds)

feature_importance_dict = model.get_fscore()
fs = ['f%i' % i for i in range(len(feature_names))]
f1 = pd.DataFrame({'f': list(feature_importance_dict.keys()),
                   'importance': list(feature_importance_dict.values())})
f2 = pd.DataFrame({'f': fs, 'feature_name': feature_names})
feature_importance = pd.merge(f1, f2, how='right', on='f')
feature_importance = feature_importance.fillna(0)
feature_importance[['feature_name', 'importance']].sort_values(by='importance', ascending=False)
ypred = model.predict(dvalid)


# t0 = dt.datetime.now()
# ytest = model.predict(dtest)
# print('Test shape OK.') if test.shape[0] == ytest.shape[0] else print('Oops')
# test['trip_duration'] = np.exp(ytest) - 1
# test[['id', 'trip_duration']].to_csv('xgb_submission.csv.gz', index=False, compression='gzip')


# print('Valid prediction mean: %.3f' % ypred.mean())
# print('Test prediction mean: %.3f' % ytest.mean())

# fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
# sns.distplot(ypred, ax=ax[0], color='blue', label='validation prediction')
# # sns.distplot(ytest, ax=ax[1], color='green', label='test prediction')
# ax[0].legend(loc=0)
# ax[1].legend(loc=0)
# plt.show()

# t1 = dt.datetime.now()
# print('Total time: %i seconds' % (t1 - t0).seconds)

