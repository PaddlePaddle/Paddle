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

import os

import numpy as np
from semi_auto_parallel_simple_net import (
    TestSimpleNetForSemiAutoParallel,
)

import paddle
import paddle.distributed as dist
from paddle import nn
from paddle.distributed import Shard
from paddle.io import DataLoader

BATCH_SIZE = 4
IMAGE_SIZE = 16
CLASS_NUM = 10


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


hook_triggered = False


def backward_hook():
    def trigger_hook(grad):
        global hook_triggered
        hook_triggered = True
        assert grad.is_dist()
        return paddle.scale(grad, 1.0)

    return trigger_hook


class MPDemoNet(nn.Layer):
    def __init__(
        self,
        param_prefix="",
    ):
        super().__init__()
        weight_attr_0 = create_numpy_like_random(param_prefix + "_0")
        weight_attr_1 = create_numpy_like_random(param_prefix + "_1")

        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self.linear_0 = nn.Linear(IMAGE_SIZE, IMAGE_SIZE, weight_attr_0)
        self.linear_1 = nn.Linear(IMAGE_SIZE, CLASS_NUM, weight_attr_1)
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

    def forward(self, x):
        self.linear_0.weight.register_hook(backward_hook())
        out = self.linear_0(x)
        # out.register_hook(backward_hook())
        out = self.relu(out)
        out = self.linear_1(out)
        return out


class TestSimpleNetWithGradientHookForSemiAutoParallel(
    TestSimpleNetForSemiAutoParallel
):
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._backend = os.getenv("backend")
        self._seed = eval(os.getenv("seed"))
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])

        paddle.set_device(self._backend)

    def create_data_loader(self):
        images = np.random.rand(BATCH_SIZE, IMAGE_SIZE).astype('float32')
        labels = np.random.rand(BATCH_SIZE, CLASS_NUM).astype('float32')
        dataset = RandomDataset(images, labels, BATCH_SIZE)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)
        return loader

    def run_static(self, layer):
        data_loader = self.create_data_loader()
        opt = paddle.optimizer.SGD(
            learning_rate=0.1, parameters=layer.parameters()
        )
        loss_fn = nn.MSELoss()

        # static training
        strategy = dist.Strategy()
        dist_model, dist_loader = dist.to_static(
            layer, data_loader, loss_fn, opt, strategy=strategy
        )
        dist_model.train()
        for step, inputs in enumerate(dist_loader()):
            input_ids, labels = inputs
            loss = dist_model(input_ids, labels)
            break

    def test_register_hook_to_static(self):
        paddle.seed(self._seed)
        np.random.seed(self._seed)

        model = MPDemoNet("mp_demo_register_hook_to_static")
        self.run_static(model)

        global hook_triggered
        assert hook_triggered
        hook_triggered = False

    def run_test_case(self):
        self.test_register_hook_to_static()


if __name__ == '__main__':
    TestSimpleNetWithGradientHookForSemiAutoParallel().run_test_case()
