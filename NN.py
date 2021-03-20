import numpy as np 
import pandas as pd 
import seaborn as sns 
sns.set(style='white')
from tqdm.notebook import tqdm 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error, r2_score

train = pd.read_csv('../mod_data/mod_train.csv')
x= train.iloc[:, :-1].values
y = train.iloc[:, -1].values.reshape(train.shape[0], 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

class Dataset_py(Dataset):
	def __init__(self, x, y):
		self.x = torch.tensor(x, dtype=torch.float32)
		self.y = torch.tensor(y, dtype=torch.float32)

	def __len__(self):
		return len(self.x)

	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]

batch_size = 500
train = Dataset_py(x_train, y_train)
test = Dataset_py(x_test, y_test)
train_dl = DataLoader(train, batch_size=batch_size)
xtrain_t = torch.tensor(x_train, dtype=torch.float32)
xtest_t = torch.tensor(x_test, dtype=torch.float32)

class MLP(nn.Module):
	def __init__(self, input_size, output_size):
		super(MLP, self).__init__()
		self.net = nn.Sequential(
		nn.Linear(input_size, 64), 
		# nn.ReLU(), 
		nn.Linear(64, 32), 
		nn.Linear(32, output_size),
		)

	def forward(self, x):
		pred = self.net(x)
		return pred 


nn_model = MLP(x_train.shape[1], 1)
cost = nn.MSELoss()
optimizer = optim.Adam(nn_model.parameters(), lr=0.1)
loss_list = []

epoch_num = 10
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
nn_y_t = nn_model(xtrain_t).detach().numpy()
test_error = np.sqrt(mean_squared_error(nn_y, y_test))
train_error = np.sqrt(mean_squared_error(nn_yt, y_train))
print('train_error, test_error', train_error, test_error)

