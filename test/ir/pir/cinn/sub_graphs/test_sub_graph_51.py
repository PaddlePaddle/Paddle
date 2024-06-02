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
# model: configs^rotate^ppyoloe_r^ppyoloe_r_crn_s_3x_dota_single_dy2st_train
# api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.ops.sigmoid||method:flatten||method:transpose||method:flatten||method:transpose||method:flatten||method:transpose
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[91, 192, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[91],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [1, 192, 64, 64], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [1, 15, 64, 64], dtype: paddle.float32, stop_gradient: False)
        var_2,  # (shape: [1, 4, 64, 64], dtype: paddle.float32, stop_gradient: False)
    ):
        var_3 = paddle.nn.functional.conv._conv_nd(
            var_0,
            self.parameter_0,
            bias=self.parameter_1,
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
        var_4 = paddle.tensor.ops.sigmoid(var_1)
        var_5 = var_4.flatten(2)
        var_6 = var_5.transpose([0, 2, 1])
        var_7 = var_2.flatten(2)
        var_8 = var_7.transpose([0, 2, 1])
        var_9 = var_3.flatten(2)
        var_10 = var_9.transpose([0, 2, 1])
        return var_3, var_4, var_6, var_8, var_10


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, 192, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1, 15, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1, 4, -1, -1),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
        ]
        self.inputs = (
            paddle.rand(shape=[1, 192, 64, 64], dtype=paddle.float32),
            paddle.rand(shape=[1, 15, 64, 64], dtype=paddle.float32),
            paddle.rand(shape=[1, 4, 64, 64], dtype=paddle.float32),
        )
        self.net = LayerCase

    # NOTE prim + cinn lead to error


if __name__ == '__main__':
    unittest.main()
