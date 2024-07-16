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
# model: ppcls^configs^ImageNet^DeiT^DeiT_tiny_distilled_patch16_224
# api:paddle.nn.functional.common.linear||api:paddle.nn.functional.common.linear||method:__add__||method:__truediv__
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[192, 1000],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[1000],
            dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
            shape=[1000],
            dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
            shape=[192, 1000],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [86, 192], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [86, 192], dtype: paddle.float32, stop_gradient: False)
    ):
        var_2 = paddle.nn.functional.common.linear(
            x=var_0, weight=self.parameter_3, bias=self.parameter_1, name=None
        )
        var_3 = paddle.nn.functional.common.linear(
            x=var_1, weight=self.parameter_0, bias=self.parameter_2, name=None
        )
        var_4 = var_2 + var_3
        var_5 = var_4 / 2
        return var_5


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
        ]
        self.inputs = (
            paddle.rand(shape=[86, 192], dtype=paddle.float32),
            paddle.rand(shape=[86, 192], dtype=paddle.float32),
        )
        self.net = LayerCase


if __name__ == '__main__':
    unittest.main()
