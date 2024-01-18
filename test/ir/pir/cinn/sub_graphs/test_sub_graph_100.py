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
# model: configs^mot^jde^jde_darknet53_30e_576x320_single_dy2st_train
# api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd
import unittest

import numpy as np

import paddle


class SIR63(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_625 = self.create_parameter(
            shape=[512],
            dtype=paddle.float32,
        )
        self.var_628 = self.create_parameter(
            shape=[512],
            dtype=paddle.float32,
        )
        self.var_624 = self.create_parameter(
            shape=[512, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_627 = self.create_parameter(
            shape=[512, 128, 3, 3],
            dtype=paddle.float32,
        )
        self.var_621 = self.create_parameter(
            shape=[512, 512, 3, 3],
            dtype=paddle.float32,
        )
        self.var_622 = self.create_parameter(
            shape=[512],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_612,  # (shape: [1, 512, 10, 18], dtype: paddle.float32, stop_gradient: False)
        var_613,  # (shape: [1, 256, 20, 36], dtype: paddle.float32, stop_gradient: False)
        var_614,  # (shape: [1, 128, 40, 72], dtype: paddle.float32, stop_gradient: False)
    ):
        var_623 = paddle.nn.functional.conv._conv_nd(
            var_612,
            self.var_621,
            bias=self.var_622,
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
        var_626 = paddle.nn.functional.conv._conv_nd(
            var_613,
            self.var_624,
            bias=self.var_625,
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
        var_629 = paddle.nn.functional.conv._conv_nd(
            var_614,
            self.var_627,
            bias=self.var_628,
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
        return var_623, var_626, var_629


class TestSIR63(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 512, 10, 18], dtype=paddle.float32),
            paddle.rand(shape=[1, 256, 20, 36], dtype=paddle.float32),
            paddle.rand(shape=[1, 128, 40, 72], dtype=paddle.float32),
        )
        self.net = SIR63()

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
