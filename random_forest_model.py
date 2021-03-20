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

input_path = '../mod_data/'
x = pd.read_csv(input_path+'train_x.csv')
y = pd.read_csv(input_path+'train_y.csv')

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

feat_labels = x.columns
forest = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1)
forest.fit(x_train, y_train)
imp_scores = forest.feature_importances_ 
indices = np.argsort(imp_scores)[::-1]
for f in range(x_train.shape[1]):
    print('%2d) %-*s %f'%(f+1, 30, feat_labels[f], imp_scores[indices[f]]))


plt.figure(figsize=(5,5))
plt.title('Feature Importance')
plt.bar(range(x_train.shape[1]), imp_scores[indices], color='lightblue', align = 'center')
plt.xticks(range(x_train.shape[1]), feat_labels, rotation=90)
plt.xlim([-1, x_train.shape[1]])
plt.tight_layout()
plt.show()