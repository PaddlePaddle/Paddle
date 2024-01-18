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
# api||paddle.tensor.manipulation.split,api||paddle.tensor.manipulation.split,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.tensor.manipulation.concat
import unittest

import numpy as np

import paddle


class SIR80(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_442 = self.create_parameter(
            shape=[160, 1, 7, 7],
            dtype=paddle.float32,
        )
        self.var_438 = self.create_parameter(
            shape=[160, 1, 3, 3],
            dtype=paddle.float32,
        )
        self.var_440 = self.create_parameter(
            shape=[160, 1, 5, 5],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_431,  # (shape: [22, 480, 14, 14], dtype: paddle.float32, stop_gradient: False)
    ):
        out = paddle.tensor.manipulation.split(var_431, [160, 160, 160], axis=1)
        var_432 = out[0]
        var_433 = out[1]
        var_434 = out[2]
        out = paddle.tensor.manipulation.split(var_431, [160, 160, 160], axis=1)
        var_435 = out[0]
        var_436 = out[1]
        var_437 = out[2]
        var_439 = paddle.nn.functional.conv._conv_nd(
            var_435,
            self.var_438,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=160,
            data_format='NCHW',
            channel_dim=1,
            op_type='depthwise_conv2d',
            use_cudnn=False,
        )
        var_441 = paddle.nn.functional.conv._conv_nd(
            var_436,
            self.var_440,
            bias=None,
            stride=[1, 1],
            padding=[2, 2],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=160,
            data_format='NCHW',
            channel_dim=1,
            op_type='depthwise_conv2d',
            use_cudnn=False,
        )
        var_443 = paddle.nn.functional.conv._conv_nd(
            var_437,
            self.var_442,
            bias=None,
            stride=[1, 1],
            padding=[3, 3],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=160,
            data_format='NCHW',
            channel_dim=1,
            op_type='depthwise_conv2d',
            use_cudnn=False,
        )
        var_444 = paddle.tensor.manipulation.concat(
            (var_439, var_441, var_443), axis=1
        )
        return var_444


class TestSIR80(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[22, 480, 14, 14], dtype=paddle.float32),
        )
        self.net = SIR80()

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
