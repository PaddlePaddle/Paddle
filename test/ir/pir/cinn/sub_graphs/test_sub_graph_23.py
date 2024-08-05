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
# model: ppcls^configs^ImageNet^EfficientNet^EfficientNetB0
# api:paddle.tensor.attribute.shape||method:__getitem__||api:paddle.tensor.random.rand||method:__radd__||api:paddle.tensor.ops.floor||api:paddle.tensor.math.multiply||method:__truediv__||api:paddle.tensor.math.add
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [11, 24, 56, 56], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [11, 24, 56, 56], dtype: paddle.float32, stop_gradient: False)
    ):
        var_3 = var_0.shape[0]
        var_4 = paddle.tensor.random.rand(shape=[var_3, 1, 1, 1])
        var_5 = 0.975 + var_4
        var_6 = paddle.tensor.ops.floor(var_5)
        var_7 = paddle.tensor.math.multiply(var_0, var_6)
        var_8 = var_7 / 0.975
        var_9 = paddle.tensor.math.add(var_8, var_1)
        return var_9


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
        ]
        self.inputs = (
            paddle.rand(shape=[11, 24, 56, 56], dtype=paddle.float32),
            paddle.rand(shape=[11, 24, 56, 56], dtype=paddle.float32),
        )
        self.net = LayerCase


if __name__ == '__main__':
    unittest.main()
