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
# api:paddle.tensor.manipulation.concat||api:paddle.nn.functional.pooling.adaptive_avg_pool2d||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.hardsigmoid||api:paddle.tensor.math.multiply
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[96],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[96, 24, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
            shape=[24, 96, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
            shape=[24],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [1, 48, 16, 16], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [1, 48, 16, 16], dtype: paddle.float32, stop_gradient: False)
    ):
        var_2 = paddle.tensor.manipulation.concat([var_1, var_0], axis=1)
        var_3 = paddle.nn.functional.pooling.adaptive_avg_pool2d(
            var_2, output_size=1, data_format='NCHW', name=None
        )
        var_4 = paddle.nn.functional.conv._conv_nd(
            var_3,
            self.parameter_2,
            bias=self.parameter_3,
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
        var_5 = paddle.nn.functional.activation.relu(var_4)
        var_6 = paddle.nn.functional.conv._conv_nd(
            var_5,
            self.parameter_1,
            bias=self.parameter_0,
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
        var_7 = paddle.nn.functional.activation.hardsigmoid(var_6)
        var_8 = paddle.tensor.math.multiply(x=var_2, y=var_7)
        return var_8


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
            paddle.rand(shape=[1, 48, 16, 16], dtype=paddle.float32),
            paddle.rand(shape=[1, 48, 16, 16], dtype=paddle.float32),
        )
        self.net = LayerCase

    def set_flags(self):
        # NOTE(Aurelius84): cinn_op.pool2d only support pool_type='avg' under adaptive=True
        paddle.set_flags({"FLAGS_deny_cinn_ops": "pool2d"})


# if __name__ == '__main__':
#     unittest.main()
