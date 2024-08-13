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
# model: ppcls^configs^ImageNet^HRNet^HRNet_W32_C
# api:paddle.nn.functional.common.upsample||api:paddle.tensor.math.add
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [10, 32, 14, 14], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [10, 32, 56, 56], dtype: paddle.float32, stop_gradient: False)
    ):
        var_2 = paddle.nn.functional.common.upsample(
            var_0, scale_factor=4, mode='nearest'
        )
        var_3 = paddle.tensor.math.add(x=var_1, y=var_2)
        return var_2, var_3


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
            paddle.rand(shape=[10, 32, 14, 14], dtype=paddle.float32),
            paddle.rand(shape=[10, 32, 56, 56], dtype=paddle.float32),
        )
        self.net = LayerCase
        self.atol = 1e-8


if __name__ == '__main__':
    unittest.main()
