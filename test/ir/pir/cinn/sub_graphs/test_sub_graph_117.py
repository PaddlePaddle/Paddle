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
# model: configs^ppyoloe^voc^ppyoloe_plus_crn_l_30e_voc_single_dy2st_train
# method||argmax,method||__mul__,method||__add__,method||flatten,method||flatten,api||paddle.tensor.manipulation.gather,method||reshape,method||__gt__,api||paddle.tensor.creation.full_like,api||paddle.tensor.search.where,method||reshape,method||flatten,api||paddle.tensor.manipulation.gather,method||reshape,api||paddle.nn.functional.input.one_hot,api||paddle.tensor.creation.to_tensor,api||paddle.tensor.search.index_select,method||__mul__,method||max,method||__mul__,method||max,method||__add__,method||__truediv__,method||__mul__,method||max,method||unsqueeze,method||__mul__
import unittest

import numpy as np

import paddle


class SIR182(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_1052,  # (shape: [1, 1, 7581], dtype: paddle.float32, stop_gradient: True)
        var_1053,  # (shape: [1, 1], dtype: paddle.int32, stop_gradient: True)
        var_1054,  # (shape: [1, 1, 1], dtype: paddle.int32, stop_gradient: True)
        var_1055,  # (shape: [1, 7581], dtype: paddle.float32, stop_gradient: True)
        var_1056,  # (shape: [1, 1, 4], dtype: paddle.float32, stop_gradient: True)
        var_1057,  # (shape: [1, 1, 7581], dtype: paddle.float32, stop_gradient: True)
        var_1058,  # (shape: [1, 1, 7581], dtype: paddle.float32, stop_gradient: True)
    ):
        var_1059 = var_1052.argmax(axis=-2)
        var_1060 = var_1053.__mul__(1)
        var_1061 = var_1059.__add__(var_1060)
        var_1062 = var_1054.flatten()
        var_1063 = var_1061.flatten()
        var_1064 = paddle.tensor.manipulation.gather(var_1062, var_1063, axis=0)
        var_1065 = var_1064.reshape([1, 7581])
        var_1066 = var_1055.__gt__(0)
        var_1067 = paddle.tensor.creation.full_like(var_1065, 20)
        var_1068 = paddle.tensor.search.where(var_1066, var_1065, var_1067)
        var_1069 = var_1056.reshape([-1, 4])
        var_1070 = var_1061.flatten()
        var_1071 = paddle.tensor.manipulation.gather(var_1069, var_1070, axis=0)
        var_1072 = var_1071.reshape([1, 7581, 4])
        var_1073 = paddle.nn.functional.input.one_hot(var_1068, 21)
        var_1074 = paddle.tensor.creation.to_tensor(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
            ]
        )
        var_1075 = paddle.tensor.search.index_select(
            var_1073, var_1074, axis=-1
        )
        var_1076 = var_1057.__mul__(var_1052)
        var_1077 = var_1076.max(axis=-1, keepdim=True)
        var_1078 = var_1058.__mul__(var_1052)
        var_1079 = var_1078.max(axis=-1, keepdim=True)
        var_1080 = var_1077.__add__(1e-09)
        var_1081 = var_1076.__truediv__(var_1080)
        var_1082 = var_1081.__mul__(var_1079)
        var_1083 = var_1082.max(-2)
        var_1084 = var_1083.unsqueeze(-1)
        var_1085 = var_1075.__mul__(var_1084)
        return var_1068, var_1072, var_1085


class TestSIR182(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 1, 7581], dtype=paddle.float32),
            paddle.randint(low=0, high=10, shape=[1, 1], dtype=paddle.int32),
            paddle.randint(low=0, high=10, shape=[1, 1, 1], dtype=paddle.int32),
            paddle.rand(shape=[1, 7581], dtype=paddle.float32),
            paddle.rand(shape=[1, 1, 4], dtype=paddle.float32),
            paddle.rand(shape=[1, 1, 7581], dtype=paddle.float32),
            paddle.rand(shape=[1, 1, 7581], dtype=paddle.float32),
        )
        self.net = SIR182()

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
