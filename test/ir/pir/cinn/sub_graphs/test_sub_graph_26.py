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
# model: ppcls^configs^ImageNet^PeleeNet^PeleeNet
# api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.norm.batch_norm||api:paddle.nn.functional.activation.relu
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
            shape=[32, 3, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [86, 3, 224, 224], dtype: paddle.float32, stop_gradient: True)
    ):
        var_1 = paddle.nn.functional.conv._conv_nd(
            var_0,
            self.parameter_3,
            bias=None,
            stride=[2, 2],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_2 = paddle.nn.functional.norm.batch_norm(
            var_1,
            self.parameter_4,
            self.parameter_1,
            weight=self.parameter_2,
            bias=self.parameter_0,
            training=True,
            momentum=0.9,
            epsilon=1e-05,
            data_format='NCHW',
            use_global_stats=None,
        )
        var_3 = paddle.nn.functional.activation.relu(var_2)
        return var_3


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, 3, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=True,
            )
        ]
        self.inputs = (
            paddle.randn(shape=[86, 3, 224, 224], dtype=paddle.float32),
        )
        self.net = LayerCase
        self.atol = 1e-5


if __name__ == '__main__':
    unittest.main()
