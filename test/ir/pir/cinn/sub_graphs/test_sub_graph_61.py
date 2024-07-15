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
# api:paddle.tensor.manipulation.split||api:paddle.tensor.manipulation.split||api:paddle.tensor.math.maximum||api:paddle.tensor.math.maximum||api:paddle.tensor.math.minimum||api:paddle.tensor.math.minimum||method:__sub__||method:clip||method:__sub__||method:clip||method:__mul__||method:__sub__||method:__sub__||method:__mul__||method:__sub__||method:__sub__||method:__mul__||method:__add__||method:__sub__||method:__add__||method:__truediv__||api:paddle.tensor.math.minimum||api:paddle.tensor.math.minimum||api:paddle.tensor.math.maximum||api:paddle.tensor.math.maximum||method:__sub__||method:__sub__||method:__mul__||method:__add__||method:__sub__||method:__truediv__||method:__sub__||method:__rsub__||method:__mul__
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [1, 4], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [1, 4], dtype: paddle.float32, stop_gradient: True)
    ):
        var_2, var_3, var_4, var_5 = paddle.tensor.manipulation.split(
            var_0, num_or_sections=4, axis=-1
        )
        var_6, var_7, var_8, var_9 = paddle.tensor.manipulation.split(
            var_1, num_or_sections=4, axis=-1
        )
        var_10 = paddle.tensor.math.maximum(var_2, var_6)
        var_11 = paddle.tensor.math.maximum(var_3, var_7)
        var_12 = paddle.tensor.math.minimum(var_4, var_8)
        var_13 = paddle.tensor.math.minimum(var_5, var_9)
        var_14 = var_12 - var_10
        var_15 = var_14.clip(0)
        var_16 = var_13 - var_11
        var_17 = var_16.clip(0)
        var_18 = var_15 * var_17
        var_19 = var_4 - var_2
        var_20 = var_5 - var_3
        var_21 = var_19 * var_20
        var_22 = var_8 - var_6
        var_23 = var_9 - var_7
        var_24 = var_22 * var_23
        var_25 = var_21 + var_24
        var_26 = var_25 - var_18
        var_27 = var_26 + 1e-10
        var_28 = var_18 / var_27
        var_29 = paddle.tensor.math.minimum(var_2, var_6)
        var_30 = paddle.tensor.math.minimum(var_3, var_7)
        var_31 = paddle.tensor.math.maximum(var_4, var_8)
        var_32 = paddle.tensor.math.maximum(var_5, var_9)
        var_33 = var_31 - var_29
        var_34 = var_32 - var_30
        var_35 = var_33 * var_34
        var_36 = var_35 + 1e-10
        var_37 = var_36 - var_27
        var_38 = var_37 / var_36
        var_39 = var_28 - var_38
        var_40 = 1 - var_39
        var_41 = var_40 * 2.0
        return var_41


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
            paddle.rand(shape=[1, 4], dtype=paddle.float32),
            paddle.rand(shape=[1, 4], dtype=paddle.float32),
        )
        self.net = LayerCase
        self.atol = 1e-5


if __name__ == '__main__':
    unittest.main()
