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
# model: ppcls^configs^ImageNet^DLA^DLA102
# api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.conv._conv_nd
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[128, 256, 1, 1],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [22, 128, 56, 56], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [22, 128, 56, 56], dtype: paddle.float32, stop_gradient: False)
    ):
        var_2 = paddle.tensor.manipulation.concat((var_0, var_1), 1)
        var_3 = paddle.nn.functional.conv._conv_nd(
            var_2,
            self.parameter_0,
            bias=None,
            stride=[1, 1],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        return var_3


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, 128, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1, 128, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
        ]
        self.inputs = (
            paddle.rand(shape=[22, 128, 56, 56], dtype=paddle.float32),
            paddle.rand(shape=[22, 128, 56, 56], dtype=paddle.float32),
        )
        self.net = LayerCase
        self.with_train = False


if __name__ == '__main__':
    unittest.main()
