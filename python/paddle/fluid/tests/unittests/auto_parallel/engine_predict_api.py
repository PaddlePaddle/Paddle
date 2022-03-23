# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
import time
import paddle.fluid as fluid
import copy
import os
import numpy as np
import subprocess
import paddle
import paddle.nn as nn
import paddle.fluid as fluid
import paddle.static as static
import paddle.nn.functional as F
import paddle.utils as utils
from paddle.fluid import layers
from paddle.io import Dataset, IterableDataset, DataLoader
from paddle.static import InputSpec
from paddle.distributed import fleet
import paddle.distributed.auto_parallel as auto
from paddle.distributed.auto_parallel.engine import Engine

paddle.enable_static()
global_process_mesh = auto.ProcessMesh(mesh=[0, 1])
batch_size = 1
batch_num = 10
hidden_size = 1024
image_size = hidden_size

paddle.seed(44)


class MyDataset(Dataset):
    def __init__(self, num_samples):
        super(MyDataset, self).__init__()
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=image_size).astype("float32")
        return input

    def __len__(self):
        return self.num_samples


class MLPLayer(nn.Layer):
    def __init__(self,
                 hidden_size=1024,
                 intermediate_size=4 * 1024,
                 dropout_ratio=0.1,
                 initializer_range=0.02):
        super(MLPLayer, self).__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(
            mean=0.0, std=initializer_range))
        bias_attr = None

        self.linear0 = nn.Linear(
            d_model, dim_feedforward, weight_attr, bias_attr=bias_attr)
        self.linear1 = nn.Linear(
            dim_feedforward, d_model, weight_attr, bias_attr=bias_attr)
        self.linear2 = nn.Linear(d_model, 1, weight_attr, bias_attr=bias_attr)
        self.norm = nn.LayerNorm(d_model, epsilon=1e-5)
        self.dropout = nn.Dropout(dropout_ratio, mode="upscale_in_train")

    def forward(self, input):
        out = self.norm(input)
        out = self.linear0(input)
        auto.shard_tensor(
            self.linear0.weight,
            dist_attr={
                "process_mesh": global_process_mesh,
                "dims_mapping": [-1, 0]
            })
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)
        auto.shard_tensor(
            self.linear1.weight,
            dist_attr={
                "process_mesh": global_process_mesh,
                "dims_mapping": [0, -1]
            })
        out = self.dropout(out)
        out = self.linear2(out)
        return out


def train():
    mlp = MLPLayer(
        hidden_size=hidden_size,
        intermediate_size=4 * hidden_size,
        dropout_ratio=0.1,
        initializer_range=0.02)

    dataset = MyDataset(batch_num * batch_size)
    inputs_spec = InputSpec([batch_size, hidden_size], 'float32', 'x')

    dist_strategy = fleet.DistributedStrategy()
    # init parallel optimizer
    dist_strategy.semi_auto = True
    fleet.init(is_collective=True, strategy=dist_strategy)

    engine = Engine(mlp, inputs_spec=inputs_spec, strategy=dist_strategy)
    engine.prepare(mode='predict')
    engine.predict(dataset, batch_size=batch_size)


if __name__ == "__main__":
    train()
