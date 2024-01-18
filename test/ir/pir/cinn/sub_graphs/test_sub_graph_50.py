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
# model: ppcls^configs^ImageNet^DPN^DPN98
# api||paddle.tensor.manipulation.split,api||paddle.tensor.math.add,api||paddle.tensor.manipulation.concat
import unittest

import numpy as np

import paddle


class SIR130(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_745,  # (shape: [22, 1056, 14, 14], dtype: paddle.float32, stop_gradient: False)
        var_746,  # (shape: [22, 1024, 14, 14], dtype: paddle.float32, stop_gradient: False)
        var_747,  # (shape: [22, 288, 14, 14], dtype: paddle.float32, stop_gradient: False)
    ):
        out = paddle.tensor.manipulation.split(
            var_745, num_or_sections=[1024, 32], axis=1
        )
        var_748 = out[0]
        var_749 = out[1]
        var_750 = paddle.tensor.math.add(x=var_746, y=var_748)
        var_751 = paddle.tensor.manipulation.concat([var_747, var_749], axis=1)
        return var_750, var_751


class TestSIR130(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[22, 1056, 14, 14], dtype=paddle.float32),
            paddle.rand(shape=[22, 1024, 14, 14], dtype=paddle.float32),
            paddle.rand(shape=[22, 288, 14, 14], dtype=paddle.float32),
        )
        self.net = SIR130()

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
