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
# model: configs^picodet^legacy_model^picodet_s_320_coco_single_dy2st_train
# api||paddle.nn.functional.pooling.adaptive_avg_pool2d,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.relu,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.activation.hardsigmoid,api||paddle.tensor.math.multiply
import unittest

import numpy as np

import paddle


class SIR17(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_82 = self.create_parameter(
            shape=[11],
            dtype=paddle.float32,
        )
        self.var_81 = self.create_parameter(
            shape=[11, 44, 1, 1],
            dtype=paddle.float32,
        )
        self.var_85 = self.create_parameter(
            shape=[44, 11, 1, 1],
            dtype=paddle.float32,
        )
        self.var_86 = self.create_parameter(
            shape=[44],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_78,  # (shape: [1, 44, 40, 40], dtype: paddle.float32, stop_gradient: False)
    ):
        var_80 = paddle.nn.functional.pooling.adaptive_avg_pool2d(
            var_78, output_size=1, data_format='NCHW', name=None
        )
        var_83 = paddle.nn.functional.conv._conv_nd(
            var_80,
            self.var_81,
            bias=self.var_82,
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
        var_84 = paddle.nn.functional.activation.relu(var_83)
        var_87 = paddle.nn.functional.conv._conv_nd(
            var_84,
            self.var_85,
            bias=self.var_86,
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
        var_88 = paddle.nn.functional.activation.hardsigmoid(var_87)
        var_89 = paddle.tensor.math.multiply(x=var_78, y=var_88)
        return var_89


class TestSIR17(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 44, 40, 40], dtype=paddle.float32),
        )
        self.net = SIR17()

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
