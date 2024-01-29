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
# model: configs^sparse_rcnn^sparse_rcnn_r50_fpn_3x_pro100_coco_single_dy2st_train
# api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.common.interpolate||method:__add__||api:paddle.nn.functional.common.interpolate||method:__add__||api:paddle.nn.functional.common.interpolate||method:__add__||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.pooling.max_pool2d
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_4 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.parameter_5 = self.create_parameter(
            shape=[256, 256, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.parameter_7 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.parameter_8 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_9 = self.create_parameter(
            shape=[256, 1024, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_10 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.parameter_11 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.parameter_12 = self.create_parameter(
            shape=[256, 512, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_13 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.parameter_14 = self.create_parameter(
            shape=[256, 2048, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_15 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [1, 256, 192, 288], dtype: paddle.float32, stop_gradient: True)
        var_1,  # (shape: [1, 512, 96, 144], dtype: paddle.float32, stop_gradient: False)
        var_2,  # (shape: [1, 1024, 48, 72], dtype: paddle.float32, stop_gradient: False)
        var_3,  # (shape: [1, 2048, 24, 36], dtype: paddle.float32, stop_gradient: False)
    ):
        var_4 = paddle.nn.functional.conv._conv_nd(
            var_0,
            self.parameter_5,
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
        var_5 = paddle.nn.functional.conv._conv_nd(
            var_1,
            self.parameter_12,
            bias=self.parameter_4,
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
        var_6 = paddle.nn.functional.conv._conv_nd(
            var_2,
            self.parameter_9,
            bias=self.parameter_6,
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
        var_7 = paddle.nn.functional.conv._conv_nd(
            var_3,
            self.parameter_14,
            bias=self.parameter_7,
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
        var_8 = paddle.nn.functional.common.interpolate(
            var_7, scale_factor=2.0, mode='nearest'
        )
        var_9 = var_6.__add__(var_8)
        var_10 = paddle.nn.functional.common.interpolate(
            var_9, scale_factor=2.0, mode='nearest'
        )
        var_11 = var_5.__add__(var_10)
        var_12 = paddle.nn.functional.common.interpolate(
            var_11, scale_factor=2.0, mode='nearest'
        )
        var_13 = var_4.__add__(var_12)
        var_14 = paddle.nn.functional.conv._conv_nd(
            var_13,
            self.parameter_8,
            bias=self.parameter_13,
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
        var_15 = paddle.nn.functional.conv._conv_nd(
            var_11,
            self.parameter_3,
            bias=self.parameter_11,
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
        var_16 = paddle.nn.functional.conv._conv_nd(
            var_9,
            self.parameter_2,
            bias=self.parameter_10,
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
        var_17 = paddle.nn.functional.conv._conv_nd(
            var_7,
            self.parameter_1,
            bias=self.parameter_15,
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
        var_18 = paddle.nn.functional.pooling.max_pool2d(var_17, 1, stride=2)
        return var_14, var_15, var_16, var_17, var_18


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 256, 192, 288], dtype=paddle.float32),
            paddle.rand(shape=[1, 512, 96, 144], dtype=paddle.float32),
            paddle.rand(shape=[1, 1024, 48, 72], dtype=paddle.float32),
            paddle.rand(shape=[1, 2048, 24, 36], dtype=paddle.float32),
        )
        self.net = LayerCase()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        if to_static:
            paddle.set_flags({'FLAGS_prim_all': with_prim})
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(
                    net, build_strategy=build_strategy, full_graph=True
                )
            else:
                net = paddle.jit.to_static(net, full_graph=True)
        paddle.seed(123)
        outs = net(*self.inputs)
        return outs

    # NOTE prim + cinn lead to error
    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(
            self.net, to_static=True, with_prim=True, with_cinn=False
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
