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

# repo: PaddleClas
# model: ppcls^configs^ImageNet^PeleeNet^PeleeNet
# api:paddle.nn.functional.pooling.avg_pool2d||method:flatten||api:paddle.nn.functional.common.dropout||api:paddle.nn.functional.common.linear
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[1000],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[704, 1000],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [86, 704, 7, 7], dtype: paddle.float32, stop_gradient: True)
    ):
        var_1 = paddle.nn.functional.pooling.avg_pool2d(
            var_0, kernel_size=[7, 7]
        )
        var_2 = var_1.flatten(1)
        var_3 = paddle.nn.functional.common.dropout(
            var_2, p=0.05, training=True
        )
        var_4 = paddle.nn.functional.common.linear(
            x=var_3, weight=self.parameter_1, bias=self.parameter_0, name=None
        )
        return var_4


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=True,
            )
        ]
        self.inputs = (
            paddle.rand(shape=[86, 704, 7, 7], dtype=paddle.float32),
        )
        self.net = LayerCase
        self.with_precision_compare = False

    # NOTE prim + cinn lead to error


if __name__ == '__main__':
    unittest.main()
