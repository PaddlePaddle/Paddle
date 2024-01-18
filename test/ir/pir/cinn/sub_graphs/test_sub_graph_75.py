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
# model: configs^rotate^ppyoloe_r^ppyoloe_r_crn_s_3x_dota_single_dy2st_train
# api||paddle.tensor.manipulation.split,method||__mul__,method||__add__,api||paddle.nn.functional.activation.elu,method||__add__,method||__mul__,method||reshape,api||paddle.nn.functional.activation.softmax,method||matmul,api||paddle.tensor.manipulation.concat,method||detach,method||detach
import unittest

import numpy as np

import paddle


class SIR186(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_1066,  # (shape: [1, 21504, 15], dtype: paddle.float32, stop_gradient: False)
        var_1067,  # (shape: [1, 21504, 4], dtype: paddle.float32, stop_gradient: False)
        var_1068,  # (shape: [1, 21504, 91], dtype: paddle.float32, stop_gradient: False)
        var_1069,  # (shape: [1, 21504, 2], dtype: paddle.float32, stop_gradient: True)
        var_1073,  # (shape: [1, 21504, 1], dtype: paddle.float32, stop_gradient: True)
        var_1083,  # (shape: [91], dtype: paddle.float32, stop_gradient: True)
    ):
        out = paddle.tensor.manipulation.split(var_1067, 2, axis=-1)
        var_1074 = out[0]
        var_1075 = out[1]
        var_1076 = var_1074.__mul__(var_1073)
        var_1077 = var_1076.__add__(var_1069)
        var_1078 = paddle.nn.functional.activation.elu(var_1075)
        var_1079 = var_1078.__add__(1.0)
        var_1080 = var_1079.__mul__(var_1073)
        var_1081 = var_1068.reshape([1, 21504, 1, 91])
        var_1082 = paddle.nn.functional.activation.softmax(var_1081)
        var_1084 = var_1082.matmul(var_1083)
        var_1085 = paddle.tensor.manipulation.concat(
            [var_1077, var_1080, var_1084], axis=-1
        )
        var_1097 = var_1066.detach()
        var_1098 = var_1085.detach()
        return var_1097, var_1098, var_1085


class TestSIR186(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 21504, 15], dtype=paddle.float32),
            paddle.rand(shape=[1, 21504, 4], dtype=paddle.float32),
            paddle.rand(shape=[1, 21504, 91], dtype=paddle.float32),
            paddle.rand(shape=[1, 21504, 2], dtype=paddle.float32),
            paddle.rand(shape=[1, 21504, 1], dtype=paddle.float32),
            paddle.rand(shape=[91], dtype=paddle.float32),
        )
        self.net = SIR186()

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
