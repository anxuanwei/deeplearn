import numpy
import numpy as np
import torch

# list1 = [[1,2,3],[4,5,6],[7,8,9]]
#
# #所有操作用矩阵表示
# #tensor
# array1 = np.array(list1)
#
#
# print(list1)
# print(array1)
# #np.array 转列表为矩阵
# array2 = np.array([[1,2,3],[4,5,6],[7,8,9]])
#
# print(array2)
# # m rows and n columns
# print(array2.shape)
# # array1
# #axis is the 矩阵合并的轴
# print(np.concatenate((array1, array2), axis=0))
# #矩阵切片，行，列
# print(array2[:,2:3])
#
# #取第0列和第二列
# idx = [0,2]
#
# print(array2[:,idx])

x = torch.tensor(3.0)
#计算x的梯度
x.requires_grad_(True)

print(x)

y = x**2

y.backward()

print(x.grad)

#画图前要把x取下来
#get x out from tensor
x = x.detach()

# width is 100 and highth = 4 , tensor
tensor1 = torch.ones((100,4))
print(tensor1)
tensor2 = torch.zeros((100,4))
print(tensor2)
tensor3 = torch.normal(0,1,(100,4))
print(tensor3)

#dim决定是在行, 列, 还是整个矩阵的求和
#keepdim保持矩阵求和后的行列不变
print(torch.sum(tensor1,dim=1,keepdim=True))

print(tensor1.shape)