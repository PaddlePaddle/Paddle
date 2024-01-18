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
# model: configs^vitdet^ppyoloe_vit_base_csppan_cae_36e_coco_single_dy2st_train
# api||paddle.nn.functional.norm.layer_norm,api||paddle.tensor.attribute.shape,method||__getitem__,method||__getitem__,api||paddle.tensor.creation.zeros_like,api||paddle.tensor.manipulation.concat,api||paddle.nn.functional.common.linear,method||__floordiv__,method||reshape,method||transpose,method||__getitem__,method||__getitem__,method||__getitem__,method||transpose,method||matmul,method||__mul__,api||paddle.nn.functional.activation.softmax,api||paddle.nn.functional.common.dropout,method||matmul,method||transpose,method||reshape,api||paddle.nn.functional.common.linear,api||paddle.nn.functional.common.dropout,method||__mul__
import unittest

import numpy as np

import paddle


class SIR6(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_526 = self.create_parameter(
            shape=[768],
            dtype=paddle.float32,
        )
        self.var_520 = self.create_parameter(
            shape=[768],
            dtype=paddle.float32,
        )
        self.var_547 = self.create_parameter(
            shape=[768],
            dtype=paddle.float32,
        )
        self.var_530 = self.create_parameter(
            shape=[768, 2304],
            dtype=paddle.float32,
        )
        self.var_521 = self.create_parameter(
            shape=[768],
            dtype=paddle.float32,
        )
        self.var_519 = self.create_parameter(
            shape=[768],
            dtype=paddle.float32,
        )
        self.var_527 = self.create_parameter(
            shape=[768],
            dtype=paddle.float32,
        )
        self.var_546 = self.create_parameter(
            shape=[768, 768],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_518,  # (shape: [1, 485, 768], dtype: paddle.float32, stop_gradient: False)
    ):
        var_522 = paddle.nn.functional.norm.layer_norm(
            var_518,
            normalized_shape=[768],
            weight=self.var_520,
            bias=self.var_521,
            epsilon=1e-06,
        )
        var_523 = paddle.tensor.attribute.shape(var_522)
        var_524 = var_523.__getitem__(1)
        var_525 = var_523.__getitem__(2)
        var_528 = paddle.tensor.creation.zeros_like(self.var_527)
        var_529 = paddle.tensor.manipulation.concat(
            (self.var_526, var_528, self.var_527)
        )
        var_531 = paddle.nn.functional.common.linear(
            var_522, weight=self.var_530, bias=var_529
        )
        var_532 = var_525.__floordiv__(12)
        var_533 = var_531.reshape((-1, var_524, 3, 12, var_532))
        var_534 = var_533.transpose((2, 0, 3, 1, 4))
        var_535 = var_534.__getitem__(0)
        var_536 = var_534.__getitem__(1)
        var_537 = var_534.__getitem__(2)
        var_538 = var_536.transpose((0, 1, 3, 2))
        var_539 = var_535.matmul(var_538)
        var_540 = var_539.__mul__(0.125)
        var_541 = paddle.nn.functional.activation.softmax(var_540, axis=-1)
        var_542 = paddle.nn.functional.common.dropout(
            var_541,
            p=0.0,
            axis=None,
            training=True,
            mode='upscale_in_train',
            name=None,
        )
        var_543 = var_542.matmul(var_537)
        var_544 = var_543.transpose((0, 2, 1, 3))
        var_545 = var_544.reshape((-1, var_524, var_525))
        var_548 = paddle.nn.functional.common.linear(
            x=var_545, weight=self.var_546, bias=self.var_547, name=None
        )
        var_549 = paddle.nn.functional.common.dropout(
            var_548,
            p=0.0,
            axis=None,
            training=True,
            mode='upscale_in_train',
            name=None,
        )
        var_550 = self.var_519.__mul__(var_549)
        return var_550


class TestSIR6(unittest.TestCase):
    def setUp(self):
        self.inputs = (paddle.rand(shape=[1, 485, 768], dtype=paddle.float32),)
        self.net = SIR6()

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
