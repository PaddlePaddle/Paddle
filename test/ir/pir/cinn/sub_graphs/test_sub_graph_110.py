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
# api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.common.interpolate,method||__add__,api||paddle.nn.functional.common.interpolate,method||__add__,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd
import unittest

import numpy as np

import paddle


class SIR35(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_161 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_158 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_167 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_160 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_163 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_148 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_145 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_144 = self.create_parameter(
            shape=[256, 512, 1, 1],
            dtype=paddle.float32,
        )
        self.var_169 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_151 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_166 = self.create_parameter(
            shape=[256, 2048, 3, 3],
            dtype=paddle.float32,
        )
        self.var_157 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_150 = self.create_parameter(
            shape=[256, 2048, 1, 1],
            dtype=paddle.float32,
        )
        self.var_164 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_147 = self.create_parameter(
            shape=[256, 1024, 1, 1],
            dtype=paddle.float32,
        )
        self.var_170 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_141,  # (shape: [1, 512, 128, 128], dtype: paddle.float32, stop_gradient: False)
        var_142,  # (shape: [1, 1024, 64, 64], dtype: paddle.float32, stop_gradient: False)
        var_143,  # (shape: [1, 2048, 32, 32], dtype: paddle.float32, stop_gradient: False)
    ):
        var_146 = paddle.nn.functional.conv._conv_nd(
            var_141,
            self.var_144,
            bias=self.var_145,
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
        var_149 = paddle.nn.functional.conv._conv_nd(
            var_142,
            self.var_147,
            bias=self.var_148,
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
        var_152 = paddle.nn.functional.conv._conv_nd(
            var_143,
            self.var_150,
            bias=self.var_151,
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
        var_153 = paddle.nn.functional.common.interpolate(
            var_152, scale_factor=2.0, mode='nearest'
        )
        var_154 = var_149.__add__(var_153)
        var_155 = paddle.nn.functional.common.interpolate(
            var_154, scale_factor=2.0, mode='nearest'
        )
        var_156 = var_146.__add__(var_155)
        var_159 = paddle.nn.functional.conv._conv_nd(
            var_156,
            self.var_157,
            bias=self.var_158,
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
        var_162 = paddle.nn.functional.conv._conv_nd(
            var_154,
            self.var_160,
            bias=self.var_161,
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
        var_165 = paddle.nn.functional.conv._conv_nd(
            var_152,
            self.var_163,
            bias=self.var_164,
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
        var_168 = paddle.nn.functional.conv._conv_nd(
            var_143,
            self.var_166,
            bias=self.var_167,
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
        var_171 = paddle.nn.functional.conv._conv_nd(
            var_168,
            self.var_169,
            bias=self.var_170,
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
        return var_159, var_162, var_165, var_168, var_171


class TestSIR35(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 512, 128, 128], dtype=paddle.float32),
            paddle.rand(shape=[1, 1024, 64, 64], dtype=paddle.float32),
            paddle.rand(shape=[1, 2048, 32, 32], dtype=paddle.float32),
        )
        self.net = SIR35()

    def train(self, net, to_static, with_prim=False, with_cinn=False):
        paddle.set_flags({'FLAGS_prim_all': with_prim})
        if to_static:
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(
                    net, build_strategy=build_strategy, full_graph=True
                )
            else:
                net = paddle.jit.to_static(net, full_graph=True)
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
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-8)


if __name__ == '__main__':
    unittest.main()
