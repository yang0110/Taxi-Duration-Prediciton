import pandas as pd 
import numpy as np 
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white')
result_path = '../results/'

model_list = ['lr', 'ridge', 'lasso', 'poly_lr', 'svm', 'svm_rbf', 'decision tree', 'knn', 'adaboost', 'random forest', 'nn', 'xgboost', 'lightbgm']
train_error_list = [0.6013599518650662, 0.6014082819055208, 0.6475190748543749, 0.5092668066488578, 0.4457634185693864, 0.4457634185693864, 0.49651048477502346, 0.5616639000820584, 0.4966186291973662, 0.45684375286468004, 0.38533719314699144, 0.421919, 0.384659218368034]
test_error_list=[0.5591325876082729, 0.5591592374453985, 0.6015741536508901, 0.7416791749281934, 0.4584919641417047, 0.4584919641417047, 0.5052637468808265, 0.6056340034203053, 0.5209706906928969, 0.4771346913889011, 0.3952695048116297, 0.4176672330467144, 0.3940023085171282]

plt.figure(figsize=(6,4))
plt.bar(np.arange(len(model_list)), train_error_list, width=0.2,  color='lightblue', align='center', label='Train')
plt.bar(np.arange(len(model_list))-0.2, test_error_list, width=0.2, color='y', align='center', label='Test')
plt.xticks(np.arange(len(model_list)), model_list, rotation=90)
# plt.xlim([-1, x_train.shape[1]])
plt.xlabel('Models', fontsize=12)
plt.ylabel('Error', fontsize=12)
plt.legend(loc=0, fontsize=12)
plt.tight_layout()
plt.savefig(result_path+'fine_tune_models_error'+'.png', dpi=100)
plt.show()