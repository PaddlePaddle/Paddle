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
# model: configs^smalldet^ppyoloe_plus_sod_crn_l_80e_coco_single_dy2st_train
# api||paddle.nn.functional.common.dropout,method||__add__,api||paddle.nn.functional.norm.layer_norm,api||paddle.nn.functional.common.linear,api||paddle.nn.functional.activation.gelu,api||paddle.nn.functional.common.dropout,api||paddle.nn.functional.common.linear,api||paddle.nn.functional.common.dropout,method||__add__,api||paddle.nn.functional.norm.layer_norm
import unittest

import numpy as np

import paddle


class SIR127(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_551 = self.create_parameter(
            shape=[1024],
            dtype=paddle.float32,
        )
        self.var_546 = self.create_parameter(
            shape=[2048],
            dtype=paddle.float32,
        )
        self.var_543 = self.create_parameter(
            shape=[1024],
            dtype=paddle.float32,
        )
        self.var_556 = self.create_parameter(
            shape=[1024],
            dtype=paddle.float32,
        )
        self.var_545 = self.create_parameter(
            shape=[1024, 2048],
            dtype=paddle.float32,
        )
        self.var_542 = self.create_parameter(
            shape=[1024],
            dtype=paddle.float32,
        )
        self.var_550 = self.create_parameter(
            shape=[2048, 1024],
            dtype=paddle.float32,
        )
        self.var_555 = self.create_parameter(
            shape=[1024],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_538,  # (shape: [1, 324, 1024], dtype: paddle.float32, stop_gradient: False)
        var_539,  # (shape: [1, 324, 1024], dtype: paddle.float32, stop_gradient: False)
    ):
        var_540 = paddle.nn.functional.common.dropout(
            var_538,
            p=0.1,
            axis=None,
            training=True,
            mode='upscale_in_train',
            name=None,
        )
        var_541 = var_539.__add__(var_540)
        var_544 = paddle.nn.functional.norm.layer_norm(
            var_541,
            normalized_shape=[1024],
            weight=self.var_542,
            bias=self.var_543,
            epsilon=1e-05,
        )
        var_547 = paddle.nn.functional.common.linear(
            x=var_544, weight=self.var_545, bias=self.var_546, name=None
        )
        var_548 = paddle.nn.functional.activation.gelu(var_547)
        var_549 = paddle.nn.functional.common.dropout(
            var_548,
            p=0.1,
            axis=None,
            training=True,
            mode='upscale_in_train',
            name=None,
        )
        var_552 = paddle.nn.functional.common.linear(
            x=var_549, weight=self.var_550, bias=self.var_551, name=None
        )
        var_553 = paddle.nn.functional.common.dropout(
            var_552,
            p=0.1,
            axis=None,
            training=True,
            mode='upscale_in_train',
            name=None,
        )
        var_554 = var_544.__add__(var_553)
        var_557 = paddle.nn.functional.norm.layer_norm(
            var_554,
            normalized_shape=[1024],
            weight=self.var_555,
            bias=self.var_556,
            epsilon=1e-05,
        )
        return var_557


class TestSIR127(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 324, 1024], dtype=paddle.float32),
            paddle.rand(shape=[1, 324, 1024], dtype=paddle.float32),
        )
        self.net = SIR127()

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
