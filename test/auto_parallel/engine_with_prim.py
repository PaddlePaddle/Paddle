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
import random

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


class MyDataset(Dataset):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples

    def __getitem__(self, index):
        paddle.seed(2023)
        np.random.seed(2023)
        random.seed(2023)
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
        paddle.seed(2023)
        np.random.seed(2023)
        random.seed(2023)
        d_model = intermediate_size
        weight1 = paddle.ParamAttr(
            initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)
        )
        bias1 = paddle.ParamAttr(
            initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)
        )
        self.linear0 = nn.Linear(hidden_size, d_model, weight1, bias1)
        self.norm = nn.LayerNorm(d_model, epsilon=1e-5)
        weight2 = paddle.ParamAttr(
            initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)
        )
        bias2 = paddle.ParamAttr(
            initializer=nn.initializer.Normal(mean=0.0, std=initializer_range)
        )
        self.linear1 = nn.Linear(d_model, 1, weight2, bias2)

    def forward(self, input):
        out0 = self.linear0(input)
        out1 = self.norm(out0)
        out = self.linear1(out1)
        auto.fetch(out, "out")
        return out


def enable_pir(flag):
    paddle.set_flags({'FLAGS_enable_pir_in_executor': flag})  # for c++
    os.environ['FLAGS_enable_pir_in_executor'] = str(flag)  # for python


def enable_prim_in_dist(flag):
    os.environ['FLAGS_enable_prim_in_distribute'] = str(flag)  # for python


def get_engine(enable_prim):
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
    strategy.seed = 2023

    if enable_prim:
        enable_pir(True)
        enable_prim_in_dist(True)
        engine = auto.Engine(
            mlp, loss, optimizer, metrics=None, strategy=strategy
        )
    else:
        enable_pir(False)
        enable_prim_in_dist(False)
        engine = auto.Engine(
            mlp, loss, optimizer, metrics=None, strategy=strategy
        )

    return engine


def train_low_level():
    # init distributed env
    paddle.distributed.fleet.init(is_collective=True)
    paddle.distributed.auto_parallel.random._rng_name_to_seed.clear()
    paddle.distributed.auto_parallel.random._inited_rng_name_to_seed.clear()
    paddle.distributed.auto_parallel.parallel_manual_seed(2023)
    paddle.distributed.auto_parallel.static.dist_context.set_default_distributed_context(
        None
    )

    # generate fake dataset
    train_dataset = MyDataset(batch_num * batch_size)

    # Build normal engine
    engine_ref = get_engine(False)
    train_dataloader_ref = engine_ref.dataloader(
        train_dataset, batch_size=batch_size, mode="train"
    )
    engine_ref.prepare(mode="train")
    for data in train_dataloader_ref:
        outs_ref = engine_ref.run(data, mode="train")

    # Build prim engine
    engine = get_engine(True)
    train_dataloader = engine.dataloader(
        train_dataset, batch_size=batch_size, mode="train"
    )
    engine.prepare(mode="train")
    for data in train_dataloader:
        outs = engine.run(data, mode="train")

    # check results
    res_ref = outs_ref['fetches']['out'][0]
    res = outs['fetches']['out'][0]
    for ref, actual in zip(res_ref, res):
        np.testing.assert_allclose(ref, actual, rtol=1e-6)


if __name__ == "__main__":
    train_low_level()
