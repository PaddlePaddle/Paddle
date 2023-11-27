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


import os

import numpy as np

import paddle
from paddle import nn
from paddle.distributed.fleet import auto
from paddle.io import Dataset

paddle.enable_static()

batch_size = 2
batch_num = 10
hidden_size = 1024
image_size = hidden_size
class_num = 10
is_feed = True
is_fetch = True
my_feed_vars = []
paddle.seed(44)


class MyDataset(Dataset):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=image_size).astype("float32")
        label = np.random.randint(0, class_num - 1, dtype="int64")
        return input, label

    def __len__(self):
        return self.num_samples


class MLPLayer(nn.Layer):
    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4 * 1024,
        initializer_range=0.02,
    ):
        super().__init__()
        d_model = hidden_size
        dim_feedforward = intermediate_size
        weight_attr = paddle.ParamAttr(
            initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)
        )
        bias_attr = None

        self.linear0 = nn.Linear(
            d_model, dim_feedforward, weight_attr, bias_attr=bias_attr
        )
        self.linear1 = nn.Linear(
            dim_feedforward, d_model, weight_attr, bias_attr=bias_attr
        )
        self.norm = nn.LayerNorm(d_model, epsilon=1e-5)

    def forward(self, input):
        out0 = self.linear0(input)
        out1 = self.linear1(out0)
        out2 = self.norm(out1)
        out = nn.functional.gelu(out2)
        return out


def enable_pir(flag):
    paddle.set_flags({'FLAGS_enable_pir_in_executor': flag})  # for c++
    os.environ['FLAGS_enable_pir_in_executor'] = str(flag)  # for python


def enable_prim_in_dist(flag):
    os.environ['FLAGS_enable_prim_in_distribute'] = str(flag)  # for python


def train_low_level():
    enable_pir(True)
    enable_prim_in_dist(True)

    paddle.distributed.auto_parallel.static.dist_context.set_default_distributed_context(
        None
    )
    mlp = MLPLayer(
        hidden_size=hidden_size,
        intermediate_size=4 * hidden_size,
        initializer_range=0.02,
    )
    loss = paddle.nn.CrossEntropyLoss()
    optimizer = paddle.optimizer.Adam(
        learning_rate=0.00001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-08,
        grad_clip=None,
    )
    strategy = auto.Strategy()
    strategy.auto_mode = "semi"

    engine = auto.Engine(mlp, loss, optimizer, metrics=None, strategy=strategy)

    # Build normal dataloader
    # train
    train_dataset = MyDataset(batch_num * batch_size)
    train_dataloader = engine.dataloader(
        train_dataset, batch_size=batch_size, mode="train"
    )
    engine.prepare(mode="train")
    for data in train_dataloader:
        outs = engine.run(data, mode="train")


if __name__ == "__main__":
    train_low_level()
