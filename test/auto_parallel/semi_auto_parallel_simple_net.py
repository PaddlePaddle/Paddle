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

import numpy as np

import paddle
import paddle.distributed as dist
import paddle.optimizer as opt
from paddle import nn

BATCH_SIZE = 16
BATCH_NUM = 4
EPOCH_NUM = 4
IMAGE_SIZE = 784
CLASS_NUM = 10


# define a random dataset
class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random([IMAGE_SIZE]).astype('float32')
        label = np.random.randint(0, CLASS_NUM, (1,)).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples


# TODO(chenweihang): update to MLP Layer later
class DemoNet(nn.Layer):
    def __init__(self):
        super().__init__()
        self.w0 = self.create_parameter(shape=[IMAGE_SIZE, IMAGE_SIZE])
        self.w1 = self.create_parameter(shape=[IMAGE_SIZE, CLASS_NUM])

    def forward(self, x):
        y = paddle.matmul(x, self.w0)
        z = paddle.matmul(y, self.w1)
        return z


class DPDemoNet(nn.Layer):
    def __init__(self, mesh):
        super().__init__()
        self.w0 = self.create_parameter(shape=[IMAGE_SIZE, IMAGE_SIZE])
        self.w1 = self.create_parameter(shape=[IMAGE_SIZE, CLASS_NUM])
        self.mesh = mesh

    def forward(self, x):
        y = paddle.matmul(
            dist.shard_tensor(
                x, dist.DistAttr(mesh=self.mesh, sharding_specs=['x', None])
            ),
            self.w0,
        )
        z = paddle.matmul(y, self.w1)
        return z


class MPDemoNet(nn.Layer):
    def __init__(self, mesh):
        super().__init__()
        self.w0 = dist.shard_tensor(
            self.create_parameter(shape=[IMAGE_SIZE, IMAGE_SIZE]),
            dist.DistAttr(mesh=mesh, sharding_specs=[None, 'x']),
        )
        self.w1 = dist.shard_tensor(
            self.create_parameter(shape=[IMAGE_SIZE, CLASS_NUM]),
            dist.DistAttr(mesh=mesh, sharding_specs=['x', None]),
        )

    def forward(self, x):
        y = paddle.matmul(x, self.w0)
        z = paddle.matmul(y, self.w1)
        return z


def train_dynamic(layer):
    # create loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    sgd = opt.SGD(learning_rate=0.001, parameters=layer.parameters())
    # create data loader
    dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
    loader = paddle.io.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=2,
    )
    # train
    for epoch_id in range(EPOCH_NUM):
        for batch_id, (image, label) in enumerate(loader()):
            out = layer(image)
            loss = loss_fn(out, label)
            loss.backward()
            sgd.step()
            sgd.clear_grad()
            print(
                f"Epoch {epoch_id} batch {batch_id}: loss = {np.mean(loss.numpy())}"
            )
    return loss


class TestSimpleNetForSemiAutoParallel:
    def __init__(self):
        self._dtype = os.getenv("dtype")
        self._seeds = eval(os.getenv("seeds"))
        self._backend = os.getenv("backend")
        self._mesh = dist.ProcessMesh([0, 1], dim_names=["x"])
        self.init_single_card_net_result()

    def init_single_card_net_result(self):
        self.base_loss = train_dynamic(DemoNet())

    def test_dp_demo_net(self):
        self.dp_loss = train_dynamic(DPDemoNet(self._mesh))

    def test_mp_demo_net(self):
        self.mp_loss = train_dynamic(MPDemoNet(self._mesh))

    def run_test_case(self):
        self.test_mp_demo_net()


if __name__ == '__main__':
    TestSimpleNetForSemiAutoParallel().run_test_case()
