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
# model: configs^tood^tood_r50_fpn_1x_coco_single_dy2st_train
# api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.common.interpolate,method||__add__,api||paddle.nn.functional.common.interpolate,method||__add__,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd
import unittest

import numpy as np

import paddle


class SIR33(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_136 = self.create_parameter(
            shape=[256, 2048, 1, 1],
            dtype=paddle.float32,
        )
        self.var_131 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_153 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_150 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_146 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_149 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_143 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_147 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_156 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_152 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_144 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_157 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_130 = self.create_parameter(
            shape=[256, 512, 1, 1],
            dtype=paddle.float32,
        )
        self.var_134 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_137 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_133 = self.create_parameter(
            shape=[256, 1024, 1, 1],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_127,  # (shape: [1, 512, 100, 152], dtype: paddle.float32, stop_gradient: False)
        var_128,  # (shape: [1, 1024, 50, 76], dtype: paddle.float32, stop_gradient: False)
        var_129,  # (shape: [1, 2048, 25, 38], dtype: paddle.float32, stop_gradient: False)
    ):
        var_132 = paddle.nn.functional.conv._conv_nd(
            var_127,
            self.var_130,
            bias=self.var_131,
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
        var_135 = paddle.nn.functional.conv._conv_nd(
            var_128,
            self.var_133,
            bias=self.var_134,
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
        var_138 = paddle.nn.functional.conv._conv_nd(
            var_129,
            self.var_136,
            bias=self.var_137,
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
        var_139 = paddle.nn.functional.common.interpolate(
            var_138, scale_factor=2.0, mode='nearest'
        )
        var_140 = var_135.__add__(var_139)
        var_141 = paddle.nn.functional.common.interpolate(
            var_140, scale_factor=2.0, mode='nearest'
        )
        var_142 = var_132.__add__(var_141)
        var_145 = paddle.nn.functional.conv._conv_nd(
            var_142,
            self.var_143,
            bias=self.var_144,
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
        var_148 = paddle.nn.functional.conv._conv_nd(
            var_140,
            self.var_146,
            bias=self.var_147,
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
        var_151 = paddle.nn.functional.conv._conv_nd(
            var_138,
            self.var_149,
            bias=self.var_150,
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
        var_154 = paddle.nn.functional.conv._conv_nd(
            var_151,
            self.var_152,
            bias=self.var_153,
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
        var_155 = paddle.nn.functional.activation.relu(var_154)
        var_158 = paddle.nn.functional.conv._conv_nd(
            var_155,
            self.var_156,
            bias=self.var_157,
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
        return var_145, var_148, var_151, var_154, var_158


class TestSIR33(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 512, 100, 152], dtype=paddle.float32),
            paddle.rand(shape=[1, 1024, 50, 76], dtype=paddle.float32),
            paddle.rand(shape=[1, 2048, 25, 38], dtype=paddle.float32),
        )
        self.net = SIR33()

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
