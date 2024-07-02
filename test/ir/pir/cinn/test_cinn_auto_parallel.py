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
import paddle.nn.functional as F
from paddle.io import DataLoader, Dataset

mesh = dist.ProcessMesh([0, 1], dim_names=["x"])


class DemoDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]).astype(
            'float32'
        ), np.array([1.0])

    def __len__(self):
        return self.num_samples


class DemoLayer(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.w = dist.shard_tensor(
            paddle.create_parameter(shape=[2, 4], dtype='float32'),
            mesh,
            [dist.Shard(1)],
        )
        self.b = paddle.to_tensor([0.1, 0.2, 0.3, 0.4])

    def forward(self, x):
        y = paddle.matmul(x, self.w)
        z = F.softmax(y + self.b)
        return z


dataset = DemoDataset(10)
loader = DataLoader(dataset, batch_size=1)


def loss_fn(logits, label):
    loss = paddle.nn.MSELoss(reduction="sum")
    logits = paddle.sum(logits, axis=[1, 2])
    return loss(logits, label)


def Eval():
    layer = DemoLayer()
    dist_layer = dist.to_static(layer, loader, loss_fn)

    dist_layer.eval()
    for data in loader():
        loss = dist_layer(data[0], data[1])
        print('loss', loss, flush=1)


# NOTE(dev): AutoParallel will launch multi-process to run this file in
# gpu:0 and gpu:1, it's not easy to apply np.testing.allclose
# between @to_static and dynamic mode.
if __name__ == "__main__":
    Eval()
