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
# model: configs^mask_rcnn^mask_rcnn_r101_vd_fpn_1x_coco_single_dy2st_train
# api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.input.one_hot||method:unsqueeze||api:paddle.tensor.manipulation.expand_as||api:paddle.tensor.search.nonzero||api:paddle.tensor.manipulation.gather_nd||api:paddle.tensor.manipulation.reshape||method:cast||method:unsqueeze||api:paddle.nn.functional.loss.binary_cross_entropy_with_logits
from base import *  # noqa: F403

from paddle.static import InputSpec


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[80],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[80, 256, 1, 1],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [2, 256, 28, 28], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [2], dtype: paddle.int32, stop_gradient: True)
        var_2,  # (shape: [2, 28, 28], dtype: paddle.int32, stop_gradient: True)
        var_3,  # (shape: [2], dtype: paddle.float32, stop_gradient: True)
    ):
        var_4 = paddle.nn.functional.conv._conv_nd(
            var_0,
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
        var_5 = paddle.nn.functional.input.one_hot(var_1, 80)
        var_6 = var_5.unsqueeze([2, 3])
        var_7 = paddle.tensor.manipulation.expand_as(var_6, var_4)
        var_8 = paddle.tensor.search.nonzero(var_7)
        var_9 = paddle.tensor.manipulation.gather_nd(var_4, var_8)
        var_10 = paddle.tensor.manipulation.reshape(var_9, [2, 28, 28])
        var_11 = var_2.cast('float32')
        var_12 = var_3.unsqueeze([1, 2])
        var_13 = paddle.nn.functional.loss.binary_cross_entropy_with_logits(
            var_10, var_11, weight=var_12, reduction='mean'
        )
        return var_13


class TestLayer(TestBase):
    def init(self):
        self.input_specs = [
            InputSpec(
                shape=(-1, 256, 28, 28),
                dtype=paddle.float32,
                name=None,
                stop_gradient=False,
            ),
            InputSpec(
                shape=(-1,), dtype=paddle.int32, name=None, stop_gradient=True
            ),
            InputSpec(
                shape=(2, -1, 28),
                dtype=paddle.int32,
                name=None,
                stop_gradient=True,
            ),
            InputSpec(
                shape=(-1,), dtype=paddle.float32, name=None, stop_gradient=True
            ),
        ]
        self.inputs = (
            paddle.rand(shape=[2, 256, 28, 28], dtype=paddle.float32),
            paddle.randint(low=0, high=10, shape=[2], dtype=paddle.int32),
            paddle.randint(
                low=0, high=10, shape=[2, 28, 28], dtype=paddle.int32
            ),
            paddle.rand(shape=[2], dtype=paddle.float32),
        )
        self.net = LayerCase


if __name__ == '__main__':
    unittest.main()
