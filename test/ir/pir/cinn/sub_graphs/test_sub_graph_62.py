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

# repo: PaddleDetection
# model: configs^mot^fairmot^fairmot_dla34_30e_1088x608_airplane_single_dy2st_train
# method:__neg__||api:paddle.tensor.ops.exp||method:__mul__||method:__neg__||api:paddle.tensor.ops.exp||method:__mul__||method:__add__||method:__add__||method:__add__||method:__mul__
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
    ):
        var_2 = -self.parameter_1
        var_3 = paddle.tensor.ops.exp(var_2)
        var_4 = var_3 * var_1
        var_5 = -self.parameter_0
        var_6 = paddle.tensor.ops.exp(var_5)
        var_7 = var_6 * var_0
        var_8 = var_4 + var_7
        var_9 = self.parameter_1 + self.parameter_0
        var_10 = var_8 + var_9
        var_11 = var_10 * 0.5
        return var_11


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1,),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1,),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
        ]
        self.inputs = (
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
        )
        self.net = LayerCase
        self.atol = 1e-8


if __name__ == '__main__':
    unittest.main()
