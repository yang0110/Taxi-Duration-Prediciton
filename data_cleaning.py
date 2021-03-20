import pandas as pd 
import numpy as np 
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

input_path = '../data/' # original dataset
train = pd.read_csv(input_path+'train.csv')
test = pd.read_csv(input_path+'test.csv')

max_dur = train['trip_duration'].max() / 3600
min_dur = train['trip_duration'].min() / 3600
train['log_trip_duration'] = np.log(train['trip_duration'].values/3600 + 1)
test['log_trip_duration'] = np.log(test['trip_duration'].values/3600 + 1)

plt.figure(figsize=(5,5))
plt.hist(train['log_trip_duration'].values, bins = 500)
plt.xlabel('log trip duration')
plt.ylabel('count')
plt.show()









