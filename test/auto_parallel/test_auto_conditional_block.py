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

import numpy as np

import paddle
import paddle.nn.functional as F
from paddle.distributed.fleet import auto

batch_num = 5
batch_size = 4
hidden_size = 1024
class_num = 10


class MyDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples

    def __getitem__(self, index):
        input = np.random.uniform(size=hidden_size).astype("float32")
        label = np.random.uniform(size=hidden_size).astype("float32")
        return input, label

    def __len__(self):
        return self.num_samples


class MLPLayer(paddle.nn.Layer):
    def __init__(
        self,
        hidden_size=1024,
        intermediate_size=4 * 1024,
    ):
        super().__init__()
        param_initializer = paddle.nn.initializer.Normal(mean=0.0, std=0.02)

        self.norm = paddle.nn.LayerNorm(hidden_size, epsilon=1e-5)
        self.linear0 = paddle.nn.Linear(
            hidden_size,
            intermediate_size,
            weight_attr=paddle.ParamAttr(initializer=param_initializer),
            bias_attr=None,
        )
        self.linear1 = paddle.nn.Linear(
            intermediate_size,
            hidden_size,
            weight_attr=paddle.ParamAttr(initializer=param_initializer),
            bias_attr=None,
        )

        self._set_cache()

    def _set_cache(self):
        self.t = paddle.arange(hidden_size, dtype="float32")
        self.t.expand([batch_size, hidden_size])

    def forward(self, input):
        out = self.norm(input)
        out = self.t + out
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)
        out = self.linear1(out)

        return out


def loss_func(pred, label):
    error_cost = paddle.nn.functional.square_error_cost(pred, label)
    error_cost = error_cost[error_cost > 0].astype("float32")
    loss = paddle.mean(error_cost)
    return loss


class TestMLP(unittest.TestCase):
    def test_conditional_block(self):
        with paddle.LazyGuard():
            mlp = MLPLayer(
                hidden_size=hidden_size,
                intermediate_size=4 * hidden_size,
            )
        optimizer = paddle.optimizer.AdamW(parameters=mlp.parameters())

        strategy = auto.Strategy()
        strategy.auto_mode = "semi"

        engine = auto.Engine(mlp, loss_func, optimizer, strategy=strategy)

        train_dataset = MyDataset(batch_num * batch_size)

        outs = engine.fit(
            train_data=train_dataset, batch_size=batch_size, log_freq=1
        )


if __name__ == "__main__":
    unittest.main()
