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
# model: ppcls^configs^ImageNet^Res2Net^Res2Net50_14w_8s
# api:paddle.nn.functional.pooling.avg_pool2d||api:paddle.tensor.manipulation.concat
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_0,  # (shape: [22, 28, 28, 28], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [22, 28, 28, 28], dtype: paddle.float32, stop_gradient: False)
        var_2,  # (shape: [22, 28, 28, 28], dtype: paddle.float32, stop_gradient: False)
        var_3,  # (shape: [22, 28, 28, 28], dtype: paddle.float32, stop_gradient: False)
        var_4,  # (shape: [22, 28, 28, 28], dtype: paddle.float32, stop_gradient: False)
        var_5,  # (shape: [22, 28, 28, 28], dtype: paddle.float32, stop_gradient: False)
        var_6,  # (shape: [22, 28, 28, 28], dtype: paddle.float32, stop_gradient: False)
        var_7,  # (shape: [22, 28, 56, 56], dtype: paddle.float32, stop_gradient: False)
    ):
        var_8 = paddle.nn.functional.pooling.avg_pool2d(
            var_7,
            kernel_size=3,
            stride=2,
            padding=1,
            ceil_mode=False,
            exclusive=True,
            divisor_override=None,
            data_format='NCHW',
            name=None,
        )
        var_9 = paddle.tensor.manipulation.concat(
            [var_0, var_1, var_2, var_3, var_4, var_5, var_6, var_8], axis=1
        )
        return var_9, var_8


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
            paddle.rand(shape=[22, 28, 28, 28], dtype=paddle.float32),
            paddle.rand(shape=[22, 28, 28, 28], dtype=paddle.float32),
            paddle.rand(shape=[22, 28, 28, 28], dtype=paddle.float32),
            paddle.rand(shape=[22, 28, 28, 28], dtype=paddle.float32),
            paddle.rand(shape=[22, 28, 28, 28], dtype=paddle.float32),
            paddle.rand(shape=[22, 28, 28, 28], dtype=paddle.float32),
            paddle.rand(shape=[22, 28, 28, 28], dtype=paddle.float32),
            paddle.rand(shape=[22, 28, 56, 56], dtype=paddle.float32),
        )
        self.net = LayerCase

    def set_flags(self):
        # NOTE(Aurelius84): cinn_op.pool2d only support pool_type='avg' under adaptive=True
        paddle.set_flags({"FLAGS_deny_cinn_ops": "pool2d"})


if __name__ == '__main__':
    unittest.main()
