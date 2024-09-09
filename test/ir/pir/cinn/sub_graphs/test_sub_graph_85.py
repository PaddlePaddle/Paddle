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
# model: configs^rotate^s2anet^s2anet_1x_spine_single_dy2st_train
# method:__sub__||api:paddle.tensor.abs||method:__lt__||method:__rmul__||method:__mul__||method:__truediv__||method:__sub__||api:paddle.tensor.search.where
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [16384, 5], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [16384, 5], dtype: paddle.float32, stop_gradient: True)
    ):
        var_2 = var_0 - var_1
        var_3 = paddle.tensor.abs(var_2)
        var_4 = var_3 < 0.1111111111111111
        var_5 = 0.5 * var_3
        var_6 = var_5 * var_3
        var_7 = var_6 / 0.1111111111111111
        var_8 = var_3 - 0.05555555555555555
        var_9 = paddle.tensor.search.where(var_4, var_7, var_8)
        return var_9


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
                stop_gradient=True,
            ),
        ]
        self.inputs = (
            paddle.rand(shape=[16384, 5], dtype=paddle.float32),
            paddle.rand(shape=[16384, 5], dtype=paddle.float32),
        )
        self.net = LayerCase


if __name__ == '__main__':
    unittest.main()
