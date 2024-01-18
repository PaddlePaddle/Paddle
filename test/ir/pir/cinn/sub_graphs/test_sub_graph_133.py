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
# model: configs^rotate^fcosr^fcosr_x50_3x_dota_single_dy2st_train
# api||paddle.nn.functional.conv._conv_nd,method||__mul__,method||__mul__,api||paddle.nn.functional.conv._conv_nd,method||__mul__,method||__add__,api||paddle.nn.functional.activation.elu,method||__mul__,api||paddle.nn.functional.conv._conv_nd,api||paddle.tensor.math.divide,api||paddle.tensor.math.sign,api||paddle.tensor.layer_function_generator.abs,api||paddle.tensor.ops.floor,api||paddle.tensor.math.multiply,method||__mul__,method||__sub__,api||paddle.tensor.manipulation.concat,method||flatten,method||transpose
import unittest

import numpy as np

import paddle


class SIR56(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_568 = self.create_parameter(
            shape=[2, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_581 = self.create_parameter(
            shape=[1, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_569 = self.create_parameter(
            shape=[2],
            dtype=paddle.float32,
        )
        self.var_582 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )
        self.var_575 = self.create_parameter(
            shape=[2],
            dtype=paddle.float32,
        )
        self.var_571 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )
        self.var_574 = self.create_parameter(
            shape=[2, 256, 3, 3],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_559,  # (shape: [1, 256, 32, 32], dtype: paddle.float32, stop_gradient: False)
        var_584,  # (shape: [1], dtype: paddle.float32, stop_gradient: True)
    ):
        var_570 = paddle.nn.functional.conv._conv_nd(
            var_559,
            self.var_568,
            bias=self.var_569,
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
        var_572 = var_570.__mul__(self.var_571)
        var_573 = var_572.__mul__(32)
        var_576 = paddle.nn.functional.conv._conv_nd(
            var_559,
            self.var_574,
            bias=self.var_575,
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
        var_577 = var_576.__mul__(self.var_571)
        var_578 = var_577.__add__(1.0)
        var_579 = paddle.nn.functional.activation.elu(var_578)
        var_580 = var_579.__mul__(32)
        var_583 = paddle.nn.functional.conv._conv_nd(
            var_559,
            self.var_581,
            bias=self.var_582,
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
        var_585 = paddle.tensor.math.divide(var_583, var_584)
        var_586 = paddle.tensor.math.sign(var_585)
        var_587 = paddle.tensor.layer_function_generator.abs(var_585)
        var_588 = paddle.tensor.ops.floor(var_587)
        var_589 = paddle.tensor.math.multiply(var_586, var_588)
        var_590 = var_589.__mul__(var_584)
        var_591 = var_583.__sub__(var_590)
        var_592 = paddle.tensor.manipulation.concat(
            [var_573, var_580, var_591], axis=1
        )
        var_593 = var_592.flatten(2)
        var_594 = var_593.transpose((0, 2, 1))
        return var_573, var_580, var_591, var_592, var_594


class TestSIR56(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 256, 32, 32], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
        )
        self.net = SIR56()

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
