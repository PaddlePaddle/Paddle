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
# model: configs^faster_rcnn^faster_rcnn_swin_tiny_fpn_1x_coco_single_dy2st_train
# api:paddle.nn.functional.conv._conv_nd||method:flatten||method:transpose||api:paddle.nn.functional.norm.layer_norm||method:transpose||method:reshape||method:flatten||method:transpose||api:paddle.nn.functional.common.dropout
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
            shape=[96],
            dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
            shape=[96],
            dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
            shape=[96, 3, 4, 4],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [1, 3, 544, 736], dtype: paddle.float32, stop_gradient: True)
    ):
        var_1 = paddle.nn.functional.conv._conv_nd(
            var_0,
            self.parameter_3,
            bias=self.parameter_1,
            stride=[4, 4],
            padding=[0, 0],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=1,
            data_format='NCHW',
            channel_dim=1,
            op_type='conv2d',
            use_cudnn=True,
        )
        var_2 = var_1.flatten(2)
        var_3 = var_2.transpose([0, 2, 1])
        var_4 = paddle.nn.functional.norm.layer_norm(
            var_3,
            normalized_shape=[96],
            weight=self.parameter_0,
            bias=self.parameter_2,
            epsilon=1e-05,
        )
        var_5 = var_4.transpose([0, 2, 1])
        var_6 = var_5.reshape([-1, 96, 136, 184])
        var_7 = var_6.flatten(2)
        var_8 = var_7.transpose([0, 2, 1])
        var_9 = paddle.nn.functional.common.dropout(
            var_8,
            p=0.0,
            axis=None,
            training=True,
            mode='upscale_in_train',
            name=None,
        )
        return var_9


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
            paddle.rand(shape=[1, 3, 544, 736], dtype=paddle.float32),
        )
        self.net = LayerCase


if __name__ == '__main__':
    unittest.main()
