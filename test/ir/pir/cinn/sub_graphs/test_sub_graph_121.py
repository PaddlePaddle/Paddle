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
# model: configs^gfl^gfl_r50_fpn_1x_coco_single_dy2st_train
# api||paddle.nn.functional.conv._conv_nd,api||paddle.nn.functional.conv._conv_nd,method||__mul__
import unittest

import numpy as np

import paddle


class SIR117(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_587 = self.create_parameter(
            shape=[68, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_584 = self.create_parameter(
            shape=[80, 256, 3, 3],
            dtype=paddle.float32,
        )
        self.var_588 = self.create_parameter(
            shape=[68],
            dtype=paddle.float32,
        )
        self.var_590 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )
        self.var_585 = self.create_parameter(
            shape=[80],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_575,  # (shape: [1, 256, 13, 19], dtype: paddle.float32, stop_gradient: False)
        var_576,  # (shape: [1, 256, 13, 19], dtype: paddle.float32, stop_gradient: False)
    ):
        var_586 = paddle.nn.functional.conv._conv_nd(
            var_575,
            self.var_584,
            bias=self.var_585,
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
        var_589 = paddle.nn.functional.conv._conv_nd(
            var_576,
            self.var_587,
            bias=self.var_588,
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
        var_591 = var_589.__mul__(self.var_590)
        return var_586, var_591


class TestSIR117(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 256, 13, 19], dtype=paddle.float32),
            paddle.rand(shape=[1, 256, 13, 19], dtype=paddle.float32),
        )
        self.net = SIR117()

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
