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

# repo: PaddleClas
# model: ppcls^configs^ImageNet^CSWinTransformer^CSWinTransformer_base_384
# api||paddle.nn.functional.norm.layer_norm,api||paddle.nn.functional.common.linear,method||chunk
import unittest

import numpy as np

import paddle


class SIR5(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.var_68 = self.create_parameter(
            shape=[96],
            dtype=paddle.float32,
        )
        self.var_67 = self.create_parameter(
            shape=[96],
            dtype=paddle.float32,
        )
        self.var_71 = self.create_parameter(
            shape=[288],
            dtype=paddle.float32,
        )
        self.var_70 = self.create_parameter(
            shape=[96, 288],
            dtype=paddle.float32,
        )

    def forward(
        self,
        var_66,  # (shape: [6, 9216, 96], dtype: paddle.float32, stop_gradient: False)
    ):
        var_69 = paddle.nn.functional.norm.layer_norm(
            var_66,
            normalized_shape=[96],
            weight=self.var_67,
            bias=self.var_68,
            epsilon=1e-05,
        )
        var_72 = paddle.nn.functional.common.linear(
            x=var_69, weight=self.var_70, bias=self.var_71, name=None
        )
        out = var_72.chunk(3, axis=-1)
        var_73 = out[0]
        var_74 = out[1]
        var_75 = out[2]
        return var_73, var_74, var_75


class TestSIR5(unittest.TestCase):
    def setUp(self):
        self.inputs = (paddle.rand(shape=[6, 9216, 96], dtype=paddle.float32),)
        self.net = SIR5()

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
