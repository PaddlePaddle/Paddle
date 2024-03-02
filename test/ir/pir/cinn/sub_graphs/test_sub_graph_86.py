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
# model: configs^rotate^s2anet^s2anet_1x_spine_single_dy2st_train
# api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.common.interpolate||method:__add__||api:paddle.nn.functional.common.interpolate||method:__add__||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd
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
            shape=[256],
            dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
            shape=[256, 1024, 1, 1],
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
            shape=[256],
            dtype=paddle.float32,
        )
        self.parameter_6 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.parameter_7 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_8 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.parameter_9 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.parameter_10 = self.create_parameter(
            shape=[256, 2048, 1, 1],
            dtype=paddle.float32,
        )
        self.parameter_11 = self.create_parameter(
            shape=[256, 256, 3, 3],
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
            shape=[256, 2048, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_15 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [1, 512, 128, 128], dtype: paddle.float32, stop_gradient: False)
        var_1,  # (shape: [1, 1024, 64, 64], dtype: paddle.float32, stop_gradient: False)
        var_2,  # (shape: [1, 2048, 32, 32], dtype: paddle.float32, stop_gradient: False)
    ):
        var_3 = paddle.nn.functional.conv._conv_nd(
            var_0,
            self.parameter_12,
            bias=self.parameter_5,
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
        var_4 = paddle.nn.functional.conv._conv_nd(
            var_1,
            self.parameter_2,
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
        var_5 = paddle.nn.functional.conv._conv_nd(
            var_2,
            self.parameter_10,
            bias=self.parameter_8,
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
        var_6 = paddle.nn.functional.common.interpolate(
            var_5, scale_factor=2.0, mode='nearest'
        )
        var_7 = var_4 + var_6
        var_8 = paddle.nn.functional.common.interpolate(
            var_7, scale_factor=2.0, mode='nearest'
        )
        var_9 = var_3 + var_8
        var_10 = paddle.nn.functional.conv._conv_nd(
            var_9,
            self.parameter_15,
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
        var_11 = paddle.nn.functional.conv._conv_nd(
            var_7,
            self.parameter_11,
            bias=self.parameter_9,
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
        var_12 = paddle.nn.functional.conv._conv_nd(
            var_5,
            self.parameter_3,
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
        var_13 = paddle.nn.functional.conv._conv_nd(
            var_2,
            self.parameter_14,
            bias=self.parameter_6,
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
        var_14 = paddle.nn.functional.conv._conv_nd(
            var_13,
            self.parameter_7,
            bias=self.parameter_13,
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
        return var_10, var_11, var_12, var_13, var_14


class TestLayer(unittest.TestCase):
    def setUp(self):
        # TODO(Aurelius84): atol only satisfy 1e-4 if shape is [1, 512, 128, 128],
        # [1, 1024, 64, 64], [1, 2048, 32, 32].
        self.inputs = (
            paddle.rand(shape=[1, 512, 4, 4], dtype=paddle.float32),
            paddle.rand(shape=[1, 1024, 2, 2], dtype=paddle.float32),
            paddle.rand(shape=[1, 2048, 1, 1], dtype=paddle.float32),
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
        paddle.seed(2024)
        outs = net(*self.inputs)
        return outs

    def test_ast_prim_cinn(self):
        st_out = self.train(self.net, to_static=True)
        cinn_out = self.train(
            self.net, to_static=True, with_prim=True, with_cinn=True
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
