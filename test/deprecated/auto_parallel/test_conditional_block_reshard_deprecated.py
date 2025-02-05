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

import paddle
import paddle.nn.functional as F
from paddle import nn
from paddle.distributed.fleet import auto
from paddle.static import InputSpec


class MLPLayer(nn.Layer):
    def __init__(
        self, hidden_size=64, intermediate_size=4 * 64, initializer_range=0.02
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, epsilon=1e-5)
        self.linear0 = nn.Linear(
            hidden_size,
            intermediate_size,
            paddle.ParamAttr(
                initializer=nn.initializer.Normal(
                    mean=0.0, std=initializer_range
                )
            ),
            bias_attr=None,
        )
        self.linear1 = nn.Linear(
            intermediate_size,
            hidden_size,
            paddle.ParamAttr(
                initializer=nn.initializer.Normal(
                    mean=0.0, std=initializer_range
                )
            ),
            bias_attr=None,
        )

    def forward(self, input):
        out = self.norm(input)

        auto.shard_tensor(
            self.linear0.weight, auto.ProcessMesh([0, 1], ["x"]), [None, "x"]
        )
        out = self.linear0(out)
        out = F.gelu(out, approximate=True)

        auto.shard_tensor(
            self.linear1.weight, auto.ProcessMesh([0, 1], ["x"]), ["x", None]
        )
        out = self.linear1(out)

        if paddle.mean(out) < 2:
            out = self.norm(out)
            out = self.linear0(out)
            out = F.gelu(out, approximate=True)
            out = self.linear1(out)
        else:
            out = self.norm(out)
            out = self.linear0(out)
            out = self.linear1(out)

        return out


def loss_fn(predict, label):
    error_cost = paddle.nn.functional.square_error_cost(predict, label)
    loss = paddle.mean(error_cost)
    return loss


class TestSubblock(unittest.TestCase):
    def test_subblock(self):
        mlp = MLPLayer()

        strategy = auto.Strategy()
        strategy.auto_mode = "semi"

        engine = auto.Engine(model=mlp, loss=loss_fn, strategy=strategy)

        input_spec = InputSpec([4, 64], 'float32', 'input')
        label_spec = InputSpec([4, 1], 'float32', 'label')
        engine.prepare(
            inputs_spec=[input_spec], labels_spec=[label_spec], mode="predict"
        )


if __name__ == "__main__":
    unittest.main()
