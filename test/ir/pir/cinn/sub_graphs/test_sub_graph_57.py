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
# method||reshape,method||transpose,method||reshape
import unittest

import numpy as np

import paddle


class SIR305(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_3020,  # (shape: [4, 48, 96, 96], dtype: paddle.float32, stop_gradient: False)
    ):
        var_3021 = var_3020.reshape([4, 48, 1, 96, 96, 1])
        var_3022 = var_3021.transpose([0, 2, 4, 3, 5, 1])
        var_3023 = var_3022.reshape([-1, 96, 48])
        return var_3023


class TestSIR305(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[4, 48, 96, 96], dtype=paddle.float32),
        )
        self.net = SIR305()

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
