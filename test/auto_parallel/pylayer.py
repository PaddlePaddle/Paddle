# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.autograd import PyLayer
from paddle.io import DataLoader


class DemoPyLayer(PyLayer):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = paddle.tanh(x)

        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensor()
        grad = dy * (1 - paddle.square(x))
        return grad


class DemoNet(nn.Layer):
    def __init__(self, mesh):
        super().__init__()
        self.mesh = mesh
        self.linear1 = paddle.nn.Linear(16, 16)

    def forward(self, x):
        x = dist.shard_tensor(x, self.mesh, [dist.Shard(0)])  # shard tensor
        y = self.linear1(x)
        return DemoPyLayer.apply(y)


class RandomDataset(paddle.io.Dataset):
    def __init__(self, images, labels, num_samples):
        self.images = images
        self.labels = labels
        self.num_samples = num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return self.num_samples


def test_pylayer():
    mesh = dist.ProcessMesh([0], dim_names=['x'])
    images = np.random.rand(4, 16).astype('float32')
    labels = np.random.rand(4, 16).astype('float32')
    dataset = RandomDataset(images, labels, 4)
    loader = DataLoader(dataset, batch_size=4)

    layer = DemoNet(mesh)
    opt = paddle.optimizer.SGD(learning_rate=0.1, parameters=layer.parameters())
    mse_loss = paddle.nn.loss.MSELoss()
    epoch = 2

    # to static
    dist_model = dist.to_static(layer, loader, mse_loss, opt)
    dist_model.train()
    for batch_id, data in enumerate(loader()):
        img, label = data
        label.stop_gradient = True
        loss = dist_model(img, label)


class DemoPyLayerCustom(PyLayer):
    @staticmethod
    def forward(ctx, x, y, z):
        ctx.save_for_backward(x, y, z)
        x1 = paddle.tanh(x)
        y1 = paddle.tanh(y)
        z1 = paddle.tanh(z)
        return x1 + y1 + z1

    @staticmethod
    def backward(ctx, grad):
        x, y, z = ctx.saved_tensor()
        x_grad = grad * (1 - paddle.square(x))
        y_grad = grad * (1 - paddle.square(y))
        return x_grad, y_grad, None


class DemoNetCustom(nn.Layer):
    def __init__(self, mesh):
        super().__init__()
        self.mesh = mesh
        self.linear1 = paddle.nn.Linear(16, 16)
        self.linear2 = paddle.nn.Linear(16, 16)
        self.linear3 = paddle.nn.Linear(16, 16)

    def forward(self, x):
        x = dist.shard_tensor(x, self.mesh, [dist.Shard(0)])  # shard tensor
        x = self.linear1(x)
        y = self.linear2(x)
        z = self.linear3(x)
        z.stop_gradient = True
        out = DemoPyLayerCustom.apply(x, y, z)
        return out


def test_pylayer_custom_op():
    mesh = dist.ProcessMesh([0], dim_names=['x'])
    images = np.random.rand(4, 16).astype('float32')
    labels = np.random.rand(4, 16).astype('float32')
    dataset = RandomDataset(images, labels, 4)
    loader = DataLoader(dataset, batch_size=4)

    layer = DemoNetCustom(mesh)
    opt = paddle.optimizer.SGD(learning_rate=0.1, parameters=layer.parameters())
    mse_loss = paddle.nn.loss.MSELoss()
    epoch = 2

    # to static
    dist_model = dist.to_static(layer, loader, mse_loss, opt)
    dist_model.train()
    for batch_id, data in enumerate(loader()):
        img, label = data
        label.stop_gradient = True
        loss = dist_model(img, label)
