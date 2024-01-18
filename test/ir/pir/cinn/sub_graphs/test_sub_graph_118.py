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
# model: configs^ssd^ssd_vgg16_300_240e_voc_single_dy2st_train
# api||paddle.tensor.search.masked_select,api||paddle.tensor.search.masked_select,api||paddle.nn.functional.loss.smooth_l1_loss,method||__mul__,api||paddle.nn.functional.loss.cross_entropy,method||squeeze,method||squeeze
import unittest

import numpy as np

import paddle


class SIR37(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_1042,  # (shape: [1, 8732, 4], dtype: paddle.float32, stop_gradient: False)
        var_1043,  # (shape: [1, 8732, 4], dtype: paddle.bool, stop_gradient: True)
        var_1044,  # (shape: [1, 8732, 4], dtype: paddle.float32, stop_gradient: True)
        var_1045,  # (shape: [1, 8732, 21], dtype: paddle.float32, stop_gradient: False)
        var_1046,  # (shape: [1, 8732, 1], dtype: paddle.int64, stop_gradient: True)
    ):
        var_1047 = paddle.tensor.search.masked_select(var_1042, var_1043)
        var_1048 = paddle.tensor.search.masked_select(var_1044, var_1043)
        var_1049 = paddle.nn.functional.loss.smooth_l1_loss(
            var_1047, var_1048, reduction='sum'
        )
        var_1050 = var_1049.__mul__(1.0)
        var_1051 = paddle.nn.functional.loss.cross_entropy(
            var_1045, var_1046, reduction='none'
        )
        var_1052 = var_1051.squeeze(-1)
        var_1053 = var_1046.squeeze(-1)
        return var_1052, var_1053, var_1051, var_1050


class TestSIR37(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 8732, 4], dtype=paddle.float32),
            paddle.rand(shape=[1, 8732, 4], dtype=paddle.bool),
            paddle.rand(shape=[1, 8732, 4], dtype=paddle.float32),
            paddle.rand(shape=[1, 8732, 21], dtype=paddle.float32),
            paddle.randint(
                low=0, high=10, shape=[1, 8732, 1], dtype=paddle.int64
            ),
        )
        self.net = SIR37()

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
