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
# model: configs^picodet^legacy_model^picodet_s_320_coco_single_dy2st_train
# method:cast||method:__add__||method:cast||method:__sub__||method:cast||method:__sub__||api:paddle.nn.functional.loss.cross_entropy||method:__mul__||api:paddle.nn.functional.loss.cross_entropy||method:__mul__||method:__add__||method:__rmul__||method:__mul__||method:sum||method:__truediv__
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [24, 8], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [24], dtype: paddle.float32, stop_gradient: True)
        var_2,  # (shape: [24], dtype: paddle.float32, stop_gradient: True)
    ):
        var_3 = var_1.cast('int64')
        var_4 = var_3 + 1
        var_5 = var_4.cast('float32')
        var_6 = var_5 - var_1
        var_7 = var_3.cast('float32')
        var_8 = var_1 - var_7
        var_9 = paddle.nn.functional.loss.cross_entropy(
            var_0, var_3, reduction='none'
        )
        var_10 = var_9 * var_6
        var_11 = paddle.nn.functional.loss.cross_entropy(
            var_0, var_4, reduction='none'
        )
        var_12 = var_11 * var_8
        var_13 = var_10 + var_12
        var_14 = 0.25 * var_13
        var_15 = var_14 * var_2
        var_16 = var_15.sum()
        var_17 = var_16 / 4.0
        return var_17


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
                shape=(-1,), dtype=paddle.float32, name=None, stop_gradient=True
            ),
            InputSpec(
                shape=(-1,), dtype=paddle.float32, name=None, stop_gradient=True
            ),
        ]
        self.inputs = (
            paddle.rand(shape=[24, 8], dtype=paddle.float32),
            paddle.rand(shape=[24], dtype=paddle.float32),
            paddle.rand(shape=[24], dtype=paddle.float32),
        )
        self.net = LayerCase

    # NOTE prim + cinn lead to error


if __name__ == '__main__':
    unittest.main()
