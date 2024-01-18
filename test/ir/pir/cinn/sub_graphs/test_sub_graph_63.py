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

# repo: PaddleClas
# model: ppcls^configs^ImageNet^MixNet^MixNet_S
# api||paddle.tensor.manipulation.split,api||paddle.tensor.manipulation.split,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.tensor.manipulation.concat
import unittest

import numpy as np

import paddle


class SIR93(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_586 = self.create_parameter(
            shape=[144, 1, 9, 9],
            dtype=paddle.float32,
        )
        self.var_580 = self.create_parameter(
            shape=[144, 1, 3, 3],
            dtype=paddle.float32,
        )
        self.var_584 = self.create_parameter(
            shape=[144, 1, 7, 7],
            dtype=paddle.float32,
        )
        self.var_588 = self.create_parameter(
            shape=[144, 1, 11, 11],
            dtype=paddle.float32,
        )
        self.var_582 = self.create_parameter(
            shape=[144, 1, 5, 5],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_569,  # (shape: [22, 720, 14, 14], dtype: paddle.float32, stop_gradient: False)
    ):
        out = paddle.tensor.manipulation.split(
            var_569, [144, 144, 144, 144, 144], axis=1
        )
        var_570 = out[0]
        var_571 = out[1]
        var_572 = out[2]
        var_573 = out[3]
        var_574 = out[4]
        out = paddle.tensor.manipulation.split(
            var_569, [144, 144, 144, 144, 144], axis=1
        )
        var_575 = out[0]
        var_576 = out[1]
        var_577 = out[2]
        var_578 = out[3]
        var_579 = out[4]
        var_581 = paddle.nn.functional.conv._conv_nd(
            var_575,
            self.var_580,
            bias=None,
            stride=[2, 2],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=144,
            data_format='NCHW',
            channel_dim=1,
            op_type='depthwise_conv2d',
            use_cudnn=False,
        )
        var_583 = paddle.nn.functional.conv._conv_nd(
            var_576,
            self.var_582,
            bias=None,
            stride=[2, 2],
            padding=[2, 2],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=144,
            data_format='NCHW',
            channel_dim=1,
            op_type='depthwise_conv2d',
            use_cudnn=False,
        )
        var_585 = paddle.nn.functional.conv._conv_nd(
            var_577,
            self.var_584,
            bias=None,
            stride=[2, 2],
            padding=[3, 3],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=144,
            data_format='NCHW',
            channel_dim=1,
            op_type='depthwise_conv2d',
            use_cudnn=False,
        )
        var_587 = paddle.nn.functional.conv._conv_nd(
            var_578,
            self.var_586,
            bias=None,
            stride=[2, 2],
            padding=[4, 4],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=144,
            data_format='NCHW',
            channel_dim=1,
            op_type='depthwise_conv2d',
            use_cudnn=False,
        )
        var_589 = paddle.nn.functional.conv._conv_nd(
            var_579,
            self.var_588,
            bias=None,
            stride=[2, 2],
            padding=[5, 5],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=144,
            data_format='NCHW',
            channel_dim=1,
            op_type='depthwise_conv2d',
            use_cudnn=False,
        )
        var_590 = paddle.tensor.manipulation.concat(
            (var_581, var_583, var_585, var_587, var_589), axis=1
        )
        return var_590


class TestSIR93(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[22, 720, 14, 14], dtype=paddle.float32),
        )
        self.net = SIR93()

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
