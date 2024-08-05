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
# api:paddle.nn.functional.pooling.max_pool2d||api:paddle.nn.functional.pooling.max_pool2d||api:paddle.nn.functional.pooling.max_pool2d||api:paddle.tensor.manipulation.concat
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [1, 192, 32, 32], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1 = paddle.nn.functional.pooling.max_pool2d(
            var_0,
            kernel_size=5,
            stride=1,
            padding=2,
            return_mask=False,
            ceil_mode=False,
            data_format='NCHW',
            name=None,
        )
        var_2 = paddle.nn.functional.pooling.max_pool2d(
            var_0,
            kernel_size=9,
            stride=1,
            padding=4,
            return_mask=False,
            ceil_mode=False,
            data_format='NCHW',
            name=None,
        )
        var_3 = paddle.nn.functional.pooling.max_pool2d(
            var_0,
            kernel_size=13,
            stride=1,
            padding=6,
            return_mask=False,
            ceil_mode=False,
            data_format='NCHW',
            name=None,
        )
        var_4 = paddle.tensor.manipulation.concat(
            [var_0, var_1, var_2, var_3], axis=1
        )
        return var_4


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            )
        ]
        self.inputs = (
            paddle.rand(shape=[1, 192, 32, 32], dtype=paddle.float32),
        )
        self.net = LayerCase

    # NOTE prim + cinn lead to error


if __name__ == '__main__':
    unittest.main()
