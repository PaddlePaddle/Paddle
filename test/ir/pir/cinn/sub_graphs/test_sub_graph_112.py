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
# model: configs^mot^mcfairmot^mcfairmot_dla34_30e_1088x608_visdrone_single_dy2st_train
# api||paddle.tensor.linalg.transpose,api||paddle.tensor.manipulation.reshape,api||paddle.tensor.manipulation.unsqueeze,api||paddle.tensor.creation.full,api||paddle.tensor.manipulation.concat,api||paddle.tensor.manipulation.concat,api||paddle.tensor.manipulation.gather_nd,api||paddle.tensor.manipulation.unsqueeze,api||paddle.tensor.manipulation.expand_as,method||__gt__,api||paddle.tensor.search.masked_select,api||paddle.tensor.manipulation.reshape
import unittest

import numpy as np

import paddle


class SIR76(paddle.nn.Layer):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        var_519,  # (shape: [1, 128, 152, 272], dtype: paddle.float32, stop_gradient: False)
        var_526,  # (shape: [1, 500], dtype: paddle.int64, stop_gradient: True)
        var_529,  # (shape: [1, 500], dtype: paddle.int32, stop_gradient: True)
    ):
        var_534 = paddle.tensor.linalg.transpose(var_519, perm=[0, 2, 3, 1])
        var_535 = paddle.tensor.manipulation.reshape(
            var_534, shape=[1, -1, 128]
        )
        var_536 = paddle.tensor.manipulation.unsqueeze(var_526, 2)
        var_537 = paddle.tensor.creation.full(
            shape=[1, 500, 1], fill_value=0, dtype='int64'
        )
        var_538 = paddle.tensor.manipulation.concat([var_537], axis=0)
        var_539 = paddle.tensor.manipulation.concat(
            x=[var_538, var_536], axis=2
        )
        var_540 = paddle.tensor.manipulation.gather_nd(var_535, index=var_539)
        var_541 = paddle.tensor.manipulation.unsqueeze(var_529, axis=2)
        var_542 = paddle.tensor.manipulation.expand_as(var_541, var_540)
        var_543 = var_542.__gt__(0)
        var_544 = paddle.tensor.search.masked_select(var_540, var_543)
        var_545 = paddle.tensor.manipulation.reshape(var_544, shape=[-1, 128])
        return var_539, var_545


class TestSIR76(unittest.TestCase):
    def setUp(self):
        self.inputs = (
            paddle.rand(shape=[1, 128, 152, 272], dtype=paddle.float32),
            paddle.randint(low=0, high=10, shape=[1, 500], dtype=paddle.int64),
            paddle.randint(low=0, high=10, shape=[1, 500], dtype=paddle.int32),
        )
        self.net = SIR76()

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
