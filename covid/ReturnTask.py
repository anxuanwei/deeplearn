import torch
import numpy as np
import pandas as pd
import csv

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader



# file = pd.read_csv(train_file)      #pd.read 读文件
# print(file.head())                    #.head()前五行

#dataset下，init初始化，把所有数据放下方便取，getitem，方便取

config = {
    "lr" : 0.01,
    "batch_size" : 32,
    "epoch" : 10,
    "momentum" : 0.9,
}


class CovidDataset(Dataset):

    def __init__(self, filepath, mode):
        self.mode = mode;
        with open(filepath, mode='r') as f:
            reader = csv.reader(f)
            ori_data = list(reader)
            #去掉第一行（表头）                  [行,列]
            csv_data = np.array(ori_data[1:])[:,1:].astype(float)
        if mode == "train":
            indices = [i for i in range(len(csv_data)) if i % 5]
            self.y = torch.tensor(csv_data[indices,-1])
        elif mode == "val":
            indices = [i for i in range(len(csv_data)) if i % 5 == 0]
            self.y = torch.tensor(csv_data[indices, -1])
        elif mode == "test":
            indices = [i for i in range(len(csv_data))]
        # self.data = torch.tensor(csv_data[indices,:-1])    #取出来的数据放进张量里
        data = torch.tensor(csv_data[indices, :-1])
        self.data = (data - data.mean(dim=0, keepdim=True)) / data.std(dim=0, keepdim=True) #减去平均值，除以标准差

    def __getitem__(self, index):
        if self.mode == "train":
            return self.data[index].float(), self.y[index].float()  #转为float减少消耗
        else:
            return self.data[index].float()

    def __len__(self):
        return len(self.data)

#模拟的实现
class Net(nn.Module):
    #初始化
    def __init__(self, input):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)


    #模型的前向代码
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        if len(x.size()) > 1:
            return x.squeeze(1)

        return x

train_file = "./../covid_train/covid.train.csv"

test_file = "./../covid_test/covid.test.csv"



train_dataset = CovidDataset(train_file, "train")
val_dataset = CovidDataset(test_file, "val")
test_dataset = CovidDataset(train_file, "test")

batch = 8;

train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)    #加载dataloader

for batchX, batchY in train_loader:
    print(batchX, batchY)


device = "cude" if torch.cuda.is_available() else "cpu"
print(device)

model = Net(93).to(device)
loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])









