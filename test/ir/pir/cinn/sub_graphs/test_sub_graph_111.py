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
# model: configs^hrnet^faster_rcnn_hrnetv2p_w18_2x_coco_single_dy2st_train
# api||paddle.nn.functional.common.interpolate,api||paddle.nn.functional.common.interpolate,api||paddle.nn.functional.common.interpolate,api||paddle.tensor.manipulation.concat,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.pooling.avg_pool2d,api||paddle.nn.functional.pooling.avg_pool2d,api||paddle.nn.functional.pooling.avg_pool2d,api||paddle.nn.functional.pooling.avg_pool2d,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd
import unittest

import numpy as np

import paddle


class SIR29(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_141 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_143 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_149 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_147 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_145 = self.create_parameter(
            shape=[256, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_135 = self.create_parameter(
            shape=[256, 270, 1, 1],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_127,  # (shape: [1, 18, 176, 264], dtype: paddle.float32, stop_gradient: True)
        var_128,  # (shape: [1, 36, 88, 132], dtype: paddle.float32, stop_gradient: False)
        var_129,  # (shape: [1, 72, 44, 66], dtype: paddle.float32, stop_gradient: False)
        var_130,  # (shape: [1, 144, 22, 33], dtype: paddle.float32, stop_gradient: False)
    ):
        var_131 = paddle.nn.functional.common.interpolate(
            var_128, scale_factor=2, mode='bilinear'
        )
        var_132 = paddle.nn.functional.common.interpolate(
            var_129, scale_factor=4, mode='bilinear'
        )
        var_133 = paddle.nn.functional.common.interpolate(
            var_130, scale_factor=8, mode='bilinear'
        )
        var_134 = paddle.tensor.manipulation.concat(
            [var_127, var_131, var_132, var_133], axis=1
        )
        var_136 = paddle.nn.functional.conv._conv_nd(
            var_134,
            self.var_135,
            bias=None,
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
        var_137 = paddle.nn.functional.pooling.avg_pool2d(
            var_136, kernel_size=2, stride=2
        )
        var_138 = paddle.nn.functional.pooling.avg_pool2d(
            var_136, kernel_size=4, stride=4
        )
        var_139 = paddle.nn.functional.pooling.avg_pool2d(
            var_136, kernel_size=8, stride=8
        )
        var_140 = paddle.nn.functional.pooling.avg_pool2d(
            var_136, kernel_size=16, stride=16
        )
        var_142 = paddle.nn.functional.conv._conv_nd(
            var_136,
            self.var_141,
            bias=None,
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
        var_144 = paddle.nn.functional.conv._conv_nd(
            var_137,
            self.var_143,
            bias=None,
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
        var_146 = paddle.nn.functional.conv._conv_nd(
            var_138,
            self.var_145,
            bias=None,
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
            var_139,
            self.var_147,
            bias=None,
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
        var_150 = paddle.nn.functional.conv._conv_nd(
            var_140,
            self.var_149,
            bias=None,
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
        return var_142, var_144, var_146, var_148, var_150


class TestSIR29(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 18, 176, 264], dtype=paddle.float32),
            paddle.rand(shape=[1, 36, 88, 132], dtype=paddle.float32),
            paddle.rand(shape=[1, 72, 44, 66], dtype=paddle.float32),
            paddle.rand(shape=[1, 144, 22, 33], dtype=paddle.float32),
        )
        self.net = SIR29()

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
