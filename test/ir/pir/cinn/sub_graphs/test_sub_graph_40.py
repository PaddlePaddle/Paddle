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
# api:paddle.tensor.manipulation.split||api:paddle.tensor.manipulation.split||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.nn.functional.conv._conv_nd||api:paddle.tensor.manipulation.concat
import unittest

import numpy as np

import paddle


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.parameter_0 = self.create_parameter(
            shape=[90, 1, 9, 9],
            dtype=paddle.float32,
        )
        self.parameter_1 = self.create_parameter(
            shape=[90, 1, 3, 3],
            dtype=paddle.float32,
        )
        self.parameter_2 = self.create_parameter(
            shape=[90, 1, 5, 5],
            dtype=paddle.float32,
        )
        self.parameter_3 = self.create_parameter(
            shape=[90, 1, 7, 7],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_0,  # (shape: [22, 360, 14, 14], dtype: paddle.float32, stop_gradient: False)
    ):
        var_1, var_2, var_3, var_4 = paddle.tensor.manipulation.split(
            var_0, [90, 90, 90, 90], axis=1
        )
        var_5 = paddle.nn.functional.conv._conv_nd(
            var_1,
            self.parameter_1,
            bias=None,
            stride=[1, 1],
            padding=[1, 1],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=90,
            data_format='NCHW',
            channel_dim=1,
            op_type='depthwise_conv2d',
            use_cudnn=False,
        )
        var_6 = paddle.nn.functional.conv._conv_nd(
            var_2,
            self.parameter_2,
            bias=None,
            stride=[1, 1],
            padding=[2, 2],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=90,
            data_format='NCHW',
            channel_dim=1,
            op_type='depthwise_conv2d',
            use_cudnn=False,
        )
        var_7 = paddle.nn.functional.conv._conv_nd(
            var_3,
            self.parameter_3,
            bias=None,
            stride=[1, 1],
            padding=[3, 3],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=90,
            data_format='NCHW',
            channel_dim=1,
            op_type='depthwise_conv2d',
            use_cudnn=False,
        )
        var_8 = paddle.nn.functional.conv._conv_nd(
            var_4,
            self.parameter_0,
            bias=None,
            stride=[1, 1],
            padding=[4, 4],
            padding_algorithm='EXPLICIT',
            dilation=[1, 1],
            groups=90,
            data_format='NCHW',
            channel_dim=1,
            op_type='depthwise_conv2d',
            use_cudnn=False,
        )
        var_9 = paddle.tensor.manipulation.concat(
            (var_5, var_6, var_7, var_8),
            axis=1,
        )
        return var_9


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[22, 360, 14, 14], dtype=paddle.float32),
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
