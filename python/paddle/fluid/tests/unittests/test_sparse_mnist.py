import paddle
from paddle.vision.transforms import Compose, Normalize, ToTensor
from paddle.fluid.framework import _test_eager_guard
import time

paddle.disable_static()
#transform = Compose([Normalize(mean=[127.5],
#                               std=[127.5],
#                               data_format='CHW')])
transform = Compose([ToTensor()])
# 使用transform对数据集做归一化
print('download training data and load training data')
train_dataset = paddle.vision.datasets.MNIST(mode='train', transform=transform)
test_dataset = paddle.vision.datasets.MNIST(mode='test', transform=transform)
print('load finished')

import numpy as np
#import matplotlib.pyplot as plt
train_data0, train_label_0 = train_dataset[0][0],train_dataset[0][1]
train_data0 = train_data0.reshape([28,28])
#plt.figure(figsize=(2,2))
#plt.imshow(train_data0, cmap=plt.cm.binary)
print('train_data0 label is: ' + str(train_label_0))


import paddle
import paddle.nn.functional as F
class SparseLeNet(paddle.nn.Layer):
    def __init__(self):
        super(SparseLeNet, self).__init__()
        #self.bn = paddle.sparse.BatchNorm(1)
        self.conv1 = paddle.sparse.Conv3D(in_channels=1, out_channels=6, kernel_size=[1, 5, 5], stride=[1, 1, 1], padding=[0, 2, 2])
        self.relu1 = paddle.sparse.ReLU()
        self.pool1 = paddle.sparse.MaxPool3D(kernel_size=[1, 2, 2], stride=[1, 2, 2])
        self.conv2 = paddle.sparse.Conv3D(in_channels=6, out_channels=16, kernel_size=[1, 5, 5], stride=[1, 1, 1])
        self.relu2 = paddle.sparse.ReLU()
        self.pool2 = paddle.sparse.MaxPool3D(kernel_size=[1, 2, 2], stride=[1, 2, 2])

        self.fc1 = paddle.nn.Linear(16*5*5, 120)
        self.fc2 = paddle.nn.Linear(120, 84)
        self.fc3 = paddle.nn.Linear(84, 10)

    def forward(self, x):
        #x = self.bn(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.to_dense()

        x = paddle.flatten(x, start_axis=1, stop_axis=-1)
        x = self.fc1(x)
        x = paddle.nn.functional.relu(x)
        x = self.fc2(x)
        x = paddle.nn.functional.relu(x)
        x = self.fc3(x)
        return x

import paddle.nn.functional as F
train_loader = paddle.io.DataLoader(train_dataset, batch_size=64, shuffle=True)
# 加载训练集 batch_size 设为 64
# sparse 训练

def prepare_data(x_data):
  x_data = paddle.transpose(x_data, perm=[0, 2, 3, 1])
  x_data = paddle.reshape(x_data, [x_data.shape[0], 1, x_data.shape[1], x_data.shape[2], x_data.shape[3]])
  return x_data

def sparse_train(model):
    model.train()
    epochs = 2
    optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())
    # 用Adam作为优化函数
    for epoch in range(epochs):
        for batch_id, data in enumerate(train_loader()):
            x_data = data[0]
            y_data = data[1]
            x_data = prepare_data(x_data)
            x_data = x_data.to_sparse_coo(4)
            x_data.stop_gradient=False
            predicts = model(x_data)
            loss = F.cross_entropy(predicts, y_data)
            # 计算损失
            acc = paddle.metric.accuracy(predicts, y_data)
            loss.backward()
            if batch_id % 300 == 0:
                print("epoch: {}, batch_id: {}, loss is: {}, acc is: {}".format(epoch, batch_id, loss.numpy(), acc.numpy()))
            optim.step()
            optim.clear_grad()

test_loader = paddle.io.DataLoader(test_dataset, places=paddle.CPUPlace(), batch_size=64)
# 加载测试数据集
def test(model):
    model.eval()
    batch_size = 64
    for batch_id, data in enumerate(test_loader()):
        x_data = data[0]
        y_data = data[1]
        x_data = prepare_data(x_data)
        x_data = x_data.to_sparse_coo(4)
        predicts = model(x_data)
        # 获取预测结果
        loss = F.cross_entropy(predicts, y_data)
        acc = paddle.metric.accuracy(predicts, y_data)
        if batch_id % 20 == 0:
            print("batch_id: {}, loss is: {}, acc is: {}".format(batch_id, loss.numpy(), acc.numpy()))

with _test_eager_guard():
  sparse_model = SparseLeNet()
  print(sparse_model)

  t0 = time.time()
  sparse_train(sparse_model)
  t1 = time.time()
  print("spare time:", t1-t0)
  test(sparse_model)
  #x = paddle.randn((1, 1,28,28,1))
  #x.stop_gradient=False
  #sparse_x = x.to_sparse_coo(4)
  #print("sparse_x values shape:", sparse_x.values().shape)
  #out = sparse_model(sparse_x)
  #out.backward(out)
  #print("end")

