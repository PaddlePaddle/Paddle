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
# model: configs^rotate^fcosr^fcosr_x50_3x_dota_single_dy2st_train
# method||split,method||unsqueeze,method||__sub__,method||__sub__,method||split,method||unsqueeze,method||__sub__,method||__mul__,api||paddle.tensor.math.sum,method||__mul__,api||paddle.tensor.math.sum,method||__mul__,api||paddle.tensor.math.sum,method||sqrt,method||__mul__,api||paddle.tensor.math.sum,method||sqrt,api||paddle.tensor.math.min,method||pow,method||pow,method||__mul__,method||__add__,method||__truediv__,method||pow,method||pow,method||__mul__,method||__add__,method||__truediv__,method||__add__,method||__rmul__,api||paddle.tensor.ops.exp,method||__truediv__,method||__rmul__,method||__add__,method||__truediv__
import unittest

import numpy as np

import paddle


class SIR73(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_992,  # (shape: [1, 21824, 2], dtype: paddle.float32, stop_gradient: True)
        var_993,  # (shape: [1, 6, 5], dtype: paddle.float32, stop_gradient: True)
        var_994,  # (shape: [1, 6, 4, 2], dtype: paddle.float32, stop_gradient: True)
    ):
        out = var_994.split(4, axis=2)
        var_995 = out[0]
        var_996 = out[1]
        var_997 = out[2]
        var_998 = out[3]
        var_999 = var_992.unsqueeze(0)
        var_1000 = var_996.__sub__(var_995)
        var_1001 = var_998.__sub__(var_995)
        out = var_993.split([2, 2, 1], axis=-1)
        var_1002 = out[0]
        var_1003 = out[1]
        var_1004 = out[2]
        var_1005 = var_1002.unsqueeze(2)
        var_1006 = var_999.__sub__(var_1005)
        var_1007 = var_1006.__mul__(var_1000)
        var_1008 = paddle.tensor.math.sum(var_1007, axis=-1)
        var_1009 = var_1006.__mul__(var_1001)
        var_1010 = paddle.tensor.math.sum(var_1009, axis=-1)
        var_1011 = var_1000.__mul__(var_1000)
        var_1012 = paddle.tensor.math.sum(var_1011, axis=-1)
        var_1013 = var_1012.sqrt()
        var_1014 = var_1001.__mul__(var_1001)
        var_1015 = paddle.tensor.math.sum(var_1014, axis=-1)
        var_1016 = var_1015.sqrt()
        var_1017 = paddle.tensor.math.min(var_1003, axis=-1, keepdim=True)
        var_1018 = var_1008.pow(2)
        var_1019 = var_1013.pow(3)
        var_1020 = var_1019.__mul__(var_1017)
        var_1021 = var_1020.__add__(1e-09)
        var_1022 = var_1018.__truediv__(var_1021)
        var_1023 = var_1010.pow(2)
        var_1024 = var_1016.pow(3)
        var_1025 = var_1024.__mul__(var_1017)
        var_1026 = var_1025.__add__(1e-09)
        var_1027 = var_1023.__truediv__(var_1026)
        var_1028 = var_1022.__add__(var_1027)
        var_1029 = var_1028.__rmul__(-6.0)
        var_1030 = paddle.tensor.ops.exp(var_1029)
        var_1031 = var_1017.__truediv__(12)
        var_1032 = var_1031.__rmul__(6.283185307179586)
        var_1033 = var_1032.__add__(1e-09)
        var_1034 = var_1030.__truediv__(var_1033)
        return var_1030, var_1034


class TestSIR73(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 21824, 2], dtype=paddle.float32),
            paddle.rand(shape=[1, 6, 5], dtype=paddle.float32),
            paddle.rand(shape=[1, 6, 4, 2], dtype=paddle.float32),
        )
        self.net = SIR73()

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
