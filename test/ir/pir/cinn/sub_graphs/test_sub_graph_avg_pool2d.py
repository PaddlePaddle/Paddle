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
# model: ppcls^configs^ImageNet^ResNeSt^ResNeSt50_fast_1s1x64d
# api:paddle.nn.functional.pooling.avg_pool2d
from base import *  # noqa: F403

from paddle.static import InputSpec


class AvgPool2dCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [22, 128, 56, 56], dtype: paddle.float32, stop_gradient: True)
    ):
        var_1 = paddle.nn.functional.pooling.avg_pool2d(
            var_0,
            kernel_size=3,
            stride=2,
            padding=1,
            ceil_mode=False,
            exclusive=True,
            divisor_override=None,
            data_format='NCHW',
            name=None,
        )
        return var_1


class TestAvgPool2d(TestBase):
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
            paddle.rand(shape=[22, 128, 56, 56], dtype=paddle.float32),
        )
        self.net = AvgPool2dCase
        self.atol = 1e-8
        self.with_cinn = False

    # NOTE prim + cinn lead to error


if __name__ == '__main__':
    unittest.main()
