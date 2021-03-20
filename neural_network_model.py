import numpy as np 
import pandas as pd 
import seaborn as sns 
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
result_path = '../results/'
input_path = '../mod_data/'
x = pd.read_csv(input_path+'train_x.csv')
y = pd.read_csv(input_path+'train_y.csv')

sc = StandardScaler()
class Dataset(torch.utils.data.Dataset):
	def __init__(self, x, y):
		self.X = sc.fit_transform(x.values)
		self.Y = y.values
	def __getitem__(self, idx):
		return self.X[idx], self.Y[idx]

	def __len__(self):
		return len(self.X)

train = Dataset(x, y)

sample_num = len(train)
test_ratio = 0.3 
test_size = int(sample_num*0.3)
train_size = sample_num - test_size
print('sample num, train size, test size {}, {}, {}'.format(sample_num, train_size, test_size))
x_train, x_test = torch.utils.data.random_split(train, (train_size, test_size))
print('len(train), len(test), {}, {}'.format(len(x_train), len(x_test)))

train_loader = torch.utils.data.DataLoader(x_train, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(x_test, batch_size=64, shuffle=False)


class Net(nn.Module):
	def __init__(self, input_size, output_size):
		super(Net, self).__init__()
		self.net = nn.Sequential(
			nn.Linear(input_size, 64),
			nn.Linear(64, 32), 
			nn.Linear(32, output_size), 
			nn.ReLU(),
			)
	def forward(self, x):
		pred = self.net(x)
		pred = torch.log(pred+1)
		return pred 

learning_rate = 1e-3

model = Net(x.shape[1], 1)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

epoch_num = 300
train_loss = [ ]
for epoch in range(epoch_num):
	for i, (x_b, y_b) in enumerate(train_loader):
		x_v, y_v = x_b.float().to('cpu'), y_b.float().to('cpu')
		outputs = model(x_v)
		loss = loss_function(outputs, y_v)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	if (epoch + 1) % 10 == 0:
		train_loss.append(loss.item())
		print('epoch {}, loss: {:.5f}'.format(epoch+1, loss.item()))



test_loss = []
with torch.no_grad():
	for x_b, y_b in test_loader:
		x_v = x_b.float().to('cpu')
		outputs = model(x_v)
		loss = loss_function(outputs, y_b)
		test_loss.append(loss.item())
		print('loss{:.5f}'.format(loss.item()))


plt.figure(figsize=(5,5))
plt.plot(np.array(train_loss)[np.array(train_loss) <= 10], label='Training Loss')
plt.plot(test_loss, label='Test loss')
plt.legend(loc=1)
plt.xlabel('epoch', fontsize=12)
plt.ylabel('loss', fontsize=12)
plt.tight_layout()
plt.savefig(result_path+'nn_loss'+'.png', dpi=100)
plt.show()










