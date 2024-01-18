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
# api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.common.interpolate,method||__add__,api||paddle.nn.functional.common.interpolate,method||__add__,api||paddle.nn.functional.common.interpolate,method||__add__,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.pooling.max_pool2d
import unittest

import numpy as np

import paddle


class SIR33(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_160 = self.create_parameter(
            shape=[256, 2048, 1, 1],
            dtype=paddle.float32,
        )
        self.var_155 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_175 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_152 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_170 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_151 = self.create_parameter(
            shape=[256, 256, 1, 1],
            dtype=paddle.float32,
        )
        self.var_169 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_176 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_158 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_179 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_172 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_154 = self.create_parameter(
            shape=[256, 512, 1, 1],
            dtype=paddle.float32,
        )
        self.var_178 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_173 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_161 = self.create_parameter(
            shape=[256],
            dtype=paddle.float32,
        )
        self.var_157 = self.create_parameter(
            shape=[256, 1024, 1, 1],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_147,  # (shape: [1, 256, 120, 184], dtype: paddle.float32, stop_gradient: True)
        var_148,  # (shape: [1, 512, 60, 92], dtype: paddle.float32, stop_gradient: False)
        var_149,  # (shape: [1, 1024, 30, 46], dtype: paddle.float32, stop_gradient: False)
        var_150,  # (shape: [1, 2048, 15, 23], dtype: paddle.float32, stop_gradient: False)
    ):
        var_153 = paddle.nn.functional.conv._conv_nd(
            var_147,
            self.var_151,
            bias=self.var_152,
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
        var_156 = paddle.nn.functional.conv._conv_nd(
            var_148,
            self.var_154,
            bias=self.var_155,
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
        var_159 = paddle.nn.functional.conv._conv_nd(
            var_149,
            self.var_157,
            bias=self.var_158,
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
        var_162 = paddle.nn.functional.conv._conv_nd(
            var_150,
            self.var_160,
            bias=self.var_161,
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
        var_163 = paddle.nn.functional.common.interpolate(
            var_162, scale_factor=2.0, mode='nearest'
        )
        var_164 = var_159.__add__(var_163)
        var_165 = paddle.nn.functional.common.interpolate(
            var_164, scale_factor=2.0, mode='nearest'
        )
        var_166 = var_156.__add__(var_165)
        var_167 = paddle.nn.functional.common.interpolate(
            var_166, scale_factor=2.0, mode='nearest'
        )
        var_168 = var_153.__add__(var_167)
        var_171 = paddle.nn.functional.conv._conv_nd(
            var_168,
            self.var_169,
            bias=self.var_170,
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
        var_174 = paddle.nn.functional.conv._conv_nd(
            var_166,
            self.var_172,
            bias=self.var_173,
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
        var_177 = paddle.nn.functional.conv._conv_nd(
            var_164,
            self.var_175,
            bias=self.var_176,
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
        var_180 = paddle.nn.functional.conv._conv_nd(
            var_162,
            self.var_178,
            bias=self.var_179,
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
        var_181 = paddle.nn.functional.pooling.max_pool2d(var_180, 1, stride=2)
        return var_171, var_174, var_177, var_180, var_181


class TestSIR33(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 256, 120, 184], dtype=paddle.float32),
            paddle.rand(shape=[1, 512, 60, 92], dtype=paddle.float32),
            paddle.rand(shape=[1, 1024, 30, 46], dtype=paddle.float32),
            paddle.rand(shape=[1, 2048, 15, 23], dtype=paddle.float32),
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
