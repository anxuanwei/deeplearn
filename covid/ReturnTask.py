import time

import torch
import numpy as np
import pandas as pd
import csv

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader



# file = pd.read_csv(train_file)      #pd.read 读文件
# print(file.head())                    #.head()前五行

#dataset下，init初始化，把所有数据放下方便取，getitem，方便取



class CovidDataset(Dataset):

    def __init__(self, filepath, mode):

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
            self.mode = mode;

    def __getitem__(self, index):
        if self.mode == "test":
            return self.data[index].float()
        else:
            return self.data[index].float(), self.y[index].float()  # 转为float减少消耗


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





# for batchX, batchY in train_loader:
#     print(batchX, batchY)


def train_val(model, train_loader, val_loader, device, epochs, optimizer, loss, savepath):
    model = model.to(device)

    plt_train_loss = []
    plt_val_loss = []

    min_val_loss = 1e9

    for epoch in range(epochs):
        train_loss = 0.0
        val_loss = 0.0

        start_time = time.time()    #记录开始时间，用来计算每一轮的计算时间

        model.train()               #模型调整为训练模式

        for batchX, batchY in train_loader:
            x, target = batchX.to(device), batchY.to(device)           #取一组数据，先把它放在设备上
            pred = model(x)                         #pred是x为y做loss做的准备
            train_bat_loss = loss(pred, target)
            train_bat_loss.backward()            #梯度回传
            optimizer.step()                   #更新模型
            optimizer.zero_grad()               #梯度清零
            train_loss += train_bat_loss.detach().cpu().item()     #放在gpu上无法相加，先放在cpu上，再用item取出数值

        plt_train_loss.append(train_loss / train_loader.__len__())          #一批的trainloss加到总的loss里


        model.eval()
        with torch.no_grad():       #验证集是为了验证效果，不能计算梯度
            for batchX, batchY in val_loader:
                x, target = batchX.to(device), batchY.to(device)
                pred = model(x)
                val_bat_loss = loss(pred, target)
                #验证要回传吗？？
                # val_bat_loss.backward()
                val_loss += val_bat_loss.cpu().item()
        plt_val_loss.append(val_loss / val_loader.__len__())
        if val_loss < min_val_loss:
            torch.save(model.state_dict(), savepath)
            min_val_loss = val_loss

        print(f"[{epoch:03d}/{epochs:03d}] {time.time() - start_time:2.2f} sec(s) Trainloss: {plt_train_loss[-1]:.6f} | Valloss: {plt_val_loss[-1]:.6f} |")










config = {
    "lr" : 0.0001,
    "batch_size" : 32,
    "epochs" : 1000,
    "momentum" : 0.9,
    "save_path" : "./../covid_result/result.pth",
}

train_file = "./../covid_train/covid.train.csv"
test_file = "./../covid_test/covid.test.csv"

train_dataset = CovidDataset(train_file, "train")
val_dataset = CovidDataset(train_file, "val")
test_dataset = CovidDataset(test_file, "test")

batch = 8;

train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)    #加载dataloader
val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=True)

device = "cude" if torch.cuda.is_available() else "cpu"
print(device)

model = Net(93).to(device)
loss = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"])

train_val(model, train_loader, val_loader, device, config["epochs"], optimizer, loss, config["save_path"])











