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
# model: ppcls^configs^ImageNet^EfficientNet^EfficientNetB0
# api||paddle.nn.functional.activation.swish,api||paddle.nn.functional.pooling.adaptive_avg_pool2d,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.swish,api||paddle.nn.functional.conv._conv_nd,api||paddle.tensor.ops.sigmoid,api||paddle.tensor.math.multiply
import unittest

import numpy as np

import paddle


class SIR168(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_1082 = self.create_parameter(
            shape=[32],
            dtype=paddle.float32,
        )
        self.var_1081 = self.create_parameter(
            shape=[32, 8, 1, 1],
            dtype=paddle.float32,
        )
        self.var_1078 = self.create_parameter(
            shape=[8],
            dtype=paddle.float32,
        )
        self.var_1077 = self.create_parameter(
            shape=[8, 32, 1, 1],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_1073,  # (shape: [43, 32, 112, 112], dtype: paddle.float32, stop_gradient: True)
    ):
        var_1075 = paddle.nn.functional.activation.swish(var_1073)
        var_1076 = paddle.nn.functional.pooling.adaptive_avg_pool2d(
            var_1075, output_size=1, data_format='NCHW', name=None
        )
        var_1079 = paddle.nn.functional.conv._conv_nd(
            var_1076,
            self.var_1077,
            bias=self.var_1078,
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
        var_1080 = paddle.nn.functional.activation.swish(var_1079)
        var_1083 = paddle.nn.functional.conv._conv_nd(
            var_1080,
            self.var_1081,
            bias=self.var_1082,
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
        var_1084 = paddle.tensor.ops.sigmoid(var_1083)
        var_1085 = paddle.tensor.math.multiply(var_1075, var_1084)
        return var_1085


class TestSIR168(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[43, 32, 112, 112], dtype=paddle.float32),
        )
        self.net = SIR168()

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
