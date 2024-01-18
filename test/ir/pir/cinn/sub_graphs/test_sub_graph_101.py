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
# api||paddle.tensor.creation.to_tensor,method||__neg__,api||paddle.tensor.ops.exp,method||__mul__,method||__add__,method||__mul__,method||__neg__,api||paddle.tensor.ops.exp,method||__mul__,method||__add__,method||__mul__,method||__add__,method||__neg__,api||paddle.tensor.ops.exp,method||__mul__,method||__add__,method||__mul__,method||__add__,method||__neg__,api||paddle.tensor.ops.exp,method||__mul__,method||__add__,method||__mul__,method||__neg__,api||paddle.tensor.ops.exp,method||__mul__,method||__add__,method||__mul__,method||__add__,method||__neg__,api||paddle.tensor.ops.exp,method||__mul__,method||__add__,method||__mul__,method||__add__,method||__neg__,api||paddle.tensor.ops.exp,method||__mul__,method||__add__,method||__mul__,method||__neg__,api||paddle.tensor.ops.exp,method||__mul__,method||__add__,method||__mul__,method||__add__,method||__neg__,api||paddle.tensor.ops.exp,method||__mul__,method||__add__,method||__mul__,method||__add__,method||__radd__,method||__add__,method||__add__,method||__radd__,method||__add__,method||__add__,method||__radd__,method||__add__,method||__add__,method||__radd__,method||__add__,method||__add__
import unittest

import numpy as np

import paddle


class SIR75(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_994 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )
        self.var_981 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )
        self.var_1001 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )
        self.var_988 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )
        self.var_948 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )
        self.var_974 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )
        self.var_961 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )
        self.var_954 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )
        self.var_968 = self.create_parameter(
            shape=[1],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_938,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_939,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_940,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_941,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_942,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_943,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_944,  # (shape: [1], dtype: paddle.float32, stop_gradient: False)
        var_945,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
        var_946,  # (shape: [], dtype: paddle.float32, stop_gradient: False)
    ):
        var_947 = paddle.tensor.creation.to_tensor(21.0, dtype='float32')
        var_949 = self.var_948.__neg__()
        var_950 = paddle.tensor.ops.exp(var_949)
        var_951 = var_950.__mul__(var_938)
        var_952 = var_951.__add__(self.var_948)
        var_953 = var_952.__mul__(0.5)
        var_955 = self.var_954.__neg__()
        var_956 = paddle.tensor.ops.exp(var_955)
        var_957 = var_956.__mul__(var_941)
        var_958 = var_957.__add__(self.var_954)
        var_959 = var_958.__mul__(0.5)
        var_960 = var_953.__add__(var_959)
        var_962 = self.var_961.__neg__()
        var_963 = paddle.tensor.ops.exp(var_962)
        var_964 = var_963.__mul__(var_944)
        var_965 = var_964.__add__(self.var_961)
        var_966 = var_965.__mul__(0.5)
        var_967 = var_960.__add__(var_966)
        var_969 = self.var_968.__neg__()
        var_970 = paddle.tensor.ops.exp(var_969)
        var_971 = var_970.__mul__(var_939)
        var_972 = var_971.__add__(self.var_968)
        var_973 = var_972.__mul__(0.5)
        var_975 = self.var_974.__neg__()
        var_976 = paddle.tensor.ops.exp(var_975)
        var_977 = var_976.__mul__(var_942)
        var_978 = var_977.__add__(self.var_974)
        var_979 = var_978.__mul__(0.5)
        var_980 = var_973.__add__(var_979)
        var_982 = self.var_981.__neg__()
        var_983 = paddle.tensor.ops.exp(var_982)
        var_984 = var_983.__mul__(var_945)
        var_985 = var_984.__add__(self.var_981)
        var_986 = var_985.__mul__(0.5)
        var_987 = var_980.__add__(var_986)
        var_989 = self.var_988.__neg__()
        var_990 = paddle.tensor.ops.exp(var_989)
        var_991 = var_990.__mul__(var_940)
        var_992 = var_991.__add__(self.var_988)
        var_993 = var_992.__mul__(0.5)
        var_995 = self.var_994.__neg__()
        var_996 = paddle.tensor.ops.exp(var_995)
        var_997 = var_996.__mul__(var_943)
        var_998 = var_997.__add__(self.var_994)
        var_999 = var_998.__mul__(0.5)
        var_1000 = var_993.__add__(var_999)
        var_1002 = self.var_1001.__neg__()
        var_1003 = paddle.tensor.ops.exp(var_1002)
        var_1004 = var_1003.__mul__(var_946)
        var_1005 = var_1004.__add__(self.var_1001)
        var_1006 = var_1005.__mul__(0.5)
        var_1007 = var_1000.__add__(var_1006)
        var_1008 = var_938.__radd__(0)
        var_1009 = var_1008.__add__(var_939)
        var_1010 = var_1009.__add__(var_940)
        var_1011 = var_941.__radd__(0)
        var_1012 = var_1011.__add__(var_942)
        var_1013 = var_1012.__add__(var_943)
        var_1014 = var_944.__radd__(0)
        var_1015 = var_1014.__add__(var_945)
        var_1016 = var_1015.__add__(var_946)
        var_1017 = var_967.__radd__(0)
        var_1018 = var_1017.__add__(var_987)
        var_1019 = var_1018.__add__(var_1007)
        return var_1010, var_1013, var_1016, var_1019, var_947


class TestSIR75(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
            paddle.rand(shape=[1], dtype=paddle.float32),
        )
        self.net = SIR75()

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
