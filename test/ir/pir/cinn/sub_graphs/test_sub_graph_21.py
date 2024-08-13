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
# model: ppcls^configs^ImageNet^DeiT^DeiT_tiny_distilled_patch16_224
# api:paddle.nn.functional.norm.layer_norm||api:paddle.nn.functional.common.linear||method:reshape||method:transpose||method:__getitem__||method:__getitem__||method:__getitem__||method:transpose||method:matmul||method:__mul__||api:paddle.nn.functional.activation.softmax||api:paddle.nn.functional.common.dropout||method:matmul||method:transpose||method:reshape||api:paddle.nn.functional.common.linear||api:paddle.nn.functional.common.dropout
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[192],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[192],
            dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
            shape=[192],
            dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
            shape=[192, 576],
            dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
            shape=[576],
            dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
            shape=[192, 192],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [86, 198, 192], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1 = paddle.nn.functional.norm.layer_norm(
            var_0,
            normalized_shape=[192],
            weight=self.parameter_1,
            bias=self.parameter_0,
            epsilon=1e-06,
        )
        var_2 = paddle.nn.functional.common.linear(
            x=var_1, weight=self.parameter_3, bias=self.parameter_4, name=None
        )
        var_3 = var_2.reshape((-1, 198, 3, 3, 64))
        var_4 = var_3.transpose((2, 0, 3, 1, 4))
        var_5 = var_4[0]
        var_6 = var_4[1]
        var_7 = var_4[2]
        var_8 = var_6.transpose((0, 1, 3, 2))
        var_9 = var_5.matmul(var_8)
        var_10 = var_9 * 0.125
        var_11 = paddle.nn.functional.activation.softmax(var_10, axis=-1)
        var_12 = paddle.nn.functional.common.dropout(
            var_11,
            p=0.0,
            axis=None,
            training=True,
            mode='upscale_in_train',
            name=None,
        )
        var_13 = var_12.matmul(var_7)
        var_14 = var_13.transpose((0, 2, 1, 3))
        var_15 = var_14.reshape((-1, 198, 192))
        var_16 = paddle.nn.functional.common.linear(
            x=var_15, weight=self.parameter_5, bias=self.parameter_2, name=None
        )
        var_17 = paddle.nn.functional.common.dropout(
            var_16,
            p=0.0,
            axis=None,
            training=True,
            mode='upscale_in_train',
            name=None,
        )
        return var_17


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, 192),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            )
        ]
        self.inputs = (paddle.rand(shape=[86, 198, 192], dtype=paddle.float32),)
        self.net = LayerCase

    # NOTE output mismatch with prim


if __name__ == '__main__':
    unittest.main()
