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
# api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.manipulation.split
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[112, 96, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[112],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [1, 96, 32, 32], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1 = paddle.nn.functional.conv._conv_nd(
            var_0,
            self.parameter_0,
            bias=self.parameter_1,
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
        var_2, var_3 = paddle.tensor.manipulation.split(var_1, [80, 32], axis=1)
        return var_1, var_2, var_3


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, -1, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            )
        ]
        self.inputs = (
            paddle.rand(shape=[1, 96, 32, 32], dtype=paddle.float32),
        )
        self.net = LayerCase


# if __name__ == '__main__':
#     unittest.main()
