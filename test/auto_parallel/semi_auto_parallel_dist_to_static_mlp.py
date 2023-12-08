# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import random

import numpy as np

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import Shard
from paddle.distributed.fleet.utils import recompute
from paddle.io import DataLoader

BATCH_SIZE = 4
BATCH_NUM = 4
SEQ_LEN = 2
IMAGE_SIZE = 16
CLASS_NUM = 8


def create_numpy_like_random(name):
    return paddle.ParamAttr(
        name=name, initializer=paddle.nn.initializer.Uniform(0, 1)
    )


class RandomDataset(paddle.io.Dataset):
    def __init__(self, images, labels, num_samples):
        self.images = images
        self.labels = labels
        self.num_samples = num_samples

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return self.num_samples


class DemoNet(nn.Layer):
    def __init__(
        self,
        mesh,
        param_prefix="",
        shard_input=False,
        shard_weight=False,
        is_recompute=False,
    ):
        super().__init__()
        self._mesh = mesh
        self.shard_input = shard_input
        self.shard_weight = shard_weight
        self.is_recompute = is_recompute
        weight_attr_0 = create_numpy_like_random(param_prefix + "_0")
        weight_attr_1 = create_numpy_like_random(param_prefix + "_1")

        self.linear_0 = nn.Linear(IMAGE_SIZE, IMAGE_SIZE, weight_attr_0)
        self.linear_1 = nn.Linear(IMAGE_SIZE, CLASS_NUM, weight_attr_1)
        if shard_weight:
            self.linear_0.weight = dist.shard_tensor(
                self.linear_0.weight,
                self._mesh,
                [Shard(1)],
                stop_gradient=False,
            )
            self.linear_1.weight = dist.shard_tensor(
                self.linear_1.weight,
                self._mesh,
                [Shard(0)],
                stop_gradient=False,
            )
        self.relu = nn.ReLU()

    def _inner_forward_fn(self, x):
        out = self.linear_0(x)
        out = self.relu(out)
        out = self.linear_1(out)
        return out

    def forward(self, x):
        if self.shard_input:
            x = dist.shard_tensor(x, self._mesh, [Shard(0)])
        if self.is_recompute:
            return recompute(self._inner_forward_fn, x)
        else:
            return self._inner_forward_fn(x)


class TestSimpleNetForSemiAutoParallel:
    def __init__(self):
        self._seed = eval(os.getenv("seed"))

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)

    def create_data_loader(self):
        images = np.random.rand(BATCH_SIZE, IMAGE_SIZE).astype('float32')
        labels = np.random.rand(BATCH_SIZE, CLASS_NUM).astype('float32')
        dataset = RandomDataset(images, labels, BATCH_SIZE)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)
        return loader

    def run_dy2static(self, layer, opt, data_loader):
        # create loss
        loss_fn = nn.MSELoss()

        # static training
        dist_model, dist_loader = dist.to_static(
            layer, data_loader, loss_fn, opt
        )
        loss_list = []
        dist_model.train()
        for epoch in range(5):
            for batch_id, (image, label) in enumerate(dist_loader()):
                loss = dist_model(image, label)
                loss_list.append(loss)
        return np.array(loss_list)

    def run_dynamic(self, layer, opt, data_loader):
        # create loss
        loss_fn = nn.MSELoss()

        loss_list = []
        for _ in range(5):
            for batch_id, (image, label) in enumerate(data_loader()):
                out = layer(image)
                loss = loss_fn(out, label)
                loss_list.append(loss.numpy())
                loss.backward()

                opt.step()
                opt.clear_grad()
        return np.array(loss_list)

    def test_dp_demo_net(self):
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self.set_random_seed(self._seed)
        data_loader = self.create_data_loader()

        self.set_random_seed(self._seed)
        dy_layer = DemoNet(mesh, "dy_dp_demonet", shard_input=True)
        dy_opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=dy_layer.parameters()
        )

        self.set_random_seed(self._seed)
        dy2static_layer = DemoNet(
            mesh, "dy2static_dp_demonet", shard_input=True
        )
        dy2static_opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=dy2static_layer.parameters()
        )

        dy_losses = self.run_dynamic(dy_layer, dy_opt, data_loader)
        dy2static_losses = self.run_dy2static(
            dy2static_layer, dy2static_opt, data_loader
        )

        # Check the loss values. Different from dygraph mode, when
        # the model is trained in dy2static mode, the loss values
        # are not the average of the losses of all processes, so
        # we should get the average loss first.
        paddle.disable_static()
        pd_partial_loss = paddle.to_tensor(dy2static_losses)
        pd_loss_list = []
        dist.all_gather(pd_loss_list, pd_partial_loss)
        np_dy2static_loss_list = [loss.numpy() for loss in pd_loss_list]
        np_dy2static_loss = np.array(np_dy2static_loss_list)
        np_dy2static_loss = np.mean(np_dy2static_loss, axis=0)

        np.testing.assert_allclose(dy_losses, np_dy2static_loss, rtol=1e-6)

    def test_mp_demo_net(self):
        mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self.set_random_seed(self._seed)
        data_loader = self.create_data_loader()

        self.set_random_seed(self._seed)
        dy_layer = DemoNet(mesh, "dy_mp_demonet", shard_weight=True)
        dy_opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=dy_layer.parameters()
        )

        self.set_random_seed(self._seed)
        dy2static_layer = DemoNet(
            mesh, "dy2static_mp_demonet", shard_weight=True
        )
        dy2static_opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=dy2static_layer.parameters()
        )

        dy_losses = self.run_dynamic(dy_layer, dy_opt, data_loader)
        dy2static_losses = self.run_dy2static(
            dy2static_layer, dy2static_opt, data_loader
        )

        np.testing.assert_allclose(dy_losses, dy2static_losses, rtol=1e-6)

    def run_test_case(self):
        self.test_dp_demo_net()
        self.test_mp_demo_net()


if __name__ == '__main__':
    TestSimpleNetForSemiAutoParallel().run_test_case()
