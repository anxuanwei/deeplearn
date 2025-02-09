import numpy as np
import torch
import random

from matplotlib import pyplot as plt


def creat_data(w, b, data_num):
    x = torch.normal(0, 1, (data_num,len(w)))
    y = torch.matmul(x, w) + b    #矩阵乘法

    noise = torch.normal(0, 0.01,y.shape)      #与y同型的噪声
    y += noise

    return x, y

num = 500 #创建的数量

TrueW = torch.tensor([8.1,2,2,4])
TrueB = torch.tensor(1.1)

x,y = creat_data(TrueW, TrueB, num)

plt.scatter(x[:,0],y) #用来画散点图
# plt.show()

def data_provider(data, label, batchsize):  #每次访问提供一组数字
    length = len(label)
    indices = list(range(length))
    # random.shuffle(indices)
    for i in range(0, length, batchsize):
        get_indices = indices[i:i+batchsize]
        get_data = data[get_indices]
        get_label = label[get_indices]

        yield get_data, get_label        #有存档点的return;


batch_size = 8

#深度学习只需要关注维度的变化
for batch_x,batch_y in data_provider(x,y,batch_size):
    print(batch_x, batch_y)
    break

def fun(x, w, b):
    pred_y = torch.matmul(x, w) + b
    return pred_y

def maeLoss(pred_y, y):
    loss = torch.sum(abs(pred_y - y))/len(y)
    return loss

# 梯度下降
#前向计算梯度，回传不计算梯度
def sgd(para, lr):
    with torch.no_grad():
        for i in para:
            i -= i.grad*lr

            i.grad_zero()       #前向运行之后，将梯度归零

lr = 0.001
w_0 = torch.normal(0, 0.01, TrueW.shape, requires_grad=True)
b_0 = torch.tensor(0.01, requires_grad=True)               #计算梯度
print(w_0, b_0)
epochs = 10

for epoch in range(epochs):
    data_loss = 0
    for batch_x, batch_y in data_provider(x,y,batch_size):
        pred = fun(batch_x, w_0, b_0)
        loss = maeLoss(pred, batch_y)
        loss.backward()
        sgd([w_0, b_0],lr)

        data_loss += loss

        print("epoch %03d: loss: %.6f"%(epoch,data_loss))

print("old data:",w_0,b_0)

# print("new data:",w_0,b_0)