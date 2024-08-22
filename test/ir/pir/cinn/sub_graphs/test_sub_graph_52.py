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
# model: configs^rotate^ppyoloe_r^ppyoloe_r_crn_s_3x_dota_single_dy2st_train
# api:paddle.tensor.manipulation.split||method:__mul__||method:__add__||api:paddle.nn.functional.activation.elu||method:__add__||method:__mul__||method:reshape||api:paddle.nn.functional.activation.softmax||method:matmul||api:paddle.tensor.manipulation.concat||method:detach||method:detach
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [1, 21504, 15], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [1, 21504, 4], dtype: paddle.float32, stop_gradient: False)
        var_2,  # (shape: [1, 21504, 91], dtype: paddle.float32, stop_gradient: False)
        var_3,  # (shape: [1, 21504, 2], dtype: paddle.float32, stop_gradient: True)
        var_4,  # (shape: [1, 21504, 1], dtype: paddle.float32, stop_gradient: True)
        var_5,  # (shape: [91], dtype: paddle.float32, stop_gradient: True)
    ):
        var_6, var_7 = paddle.tensor.manipulation.split(var_1, 2, axis=-1)
        var_8 = var_6 * var_4
        var_9 = var_8 + var_3
        var_10 = paddle.nn.functional.activation.elu(var_7)
        var_11 = var_10 + 1.0
        var_12 = var_11 * var_4
        var_13 = var_2.reshape([1, 21504, 1, 91])
        var_14 = paddle.nn.functional.activation.softmax(var_13)
        var_15 = var_14.matmul(var_5)
        var_16 = paddle.tensor.manipulation.concat(
            [var_9, var_12, var_15], axis=-1
        )
        return var_0, var_16


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=True,
            ),
            InputSpec(
                shape=(-1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=True,
            ),
            InputSpec(
                shape=(-1,), dtype=paddle.float32, name=None, stop_gradient=True
            ),
        ]
        self.inputs = (
            paddle.rand(shape=[1, 21504, 15], dtype=paddle.float32),
            paddle.rand(shape=[1, 21504, 4], dtype=paddle.float32),
            paddle.rand(shape=[1, 21504, 91], dtype=paddle.float32),
            paddle.rand(shape=[1, 21504, 2], dtype=paddle.float32),
            paddle.rand(shape=[1, 21504, 1], dtype=paddle.float32),
            paddle.rand(shape=[91], dtype=paddle.float32),
        )
        self.net = LayerCase


if __name__ == '__main__':
    unittest.main()
