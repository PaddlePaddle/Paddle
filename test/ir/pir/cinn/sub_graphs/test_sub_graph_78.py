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
# model: configs^fcos^fcos_r50_fpn_iou_1x_coco_single_dy2st_train
# api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||method:__mul__||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.activation.relu
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[4],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
            shape=[1, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
            shape=[80],
            dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
            shape=[80, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
            shape=[4, 256, 3, 3],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [1, 256, 13, 19], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [1, 256, 13, 19], dtype: paddle.float32, stop_gradient: False)
    ):
        var_2 = paddle.nn.functional.conv._conv_nd(
            var_0,
            self.parameter_5,
            bias=self.parameter_4,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_3 = paddle.nn.functional.conv._conv_nd(
            var_1,
            self.parameter_6,
            bias=self.parameter_0,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_4 = var_3 * self.parameter_1
        var_5 = paddle.nn.functional.conv._conv_nd(
            var_1,
            self.parameter_3,
            bias=self.parameter_2,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_6 = paddle.nn.functional.activation.relu(var_4)
        return var_2, var_6, var_5


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, 256, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1, 256, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
        ]
        self.inputs = (
            paddle.rand(shape=[1, 256, 13, 19], dtype=paddle.float32),
            paddle.rand(shape=[1, 256, 13, 19], dtype=paddle.float32),
        )
        self.net = LayerCase


if __name__ == '__main__':
    unittest.main()
