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

import unittest

import numpy as np

import paddle
from paddle.base import core


class LayerCase(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.relu = paddle.nn.functional.relu

    def triple_full(self):
        y1 = paddle.full([4], 1)
        y2 = paddle.full([4], 0)
        y3 = paddle.full([4], 0)
        return y1, y2, y3

    def concat_case_1(self):
        y1, y2, y3 = self.triple_full()
        out = paddle.concat([y1, y2, y3])
        return self.relu(out)

    def concat_case_2(self):
        y1, y2, y3 = self.triple_full()
        out = paddle.concat([y2, y1, y3])
        return self.relu(out)

    def concat_case_3(self):
        y1, y2, y3 = self.triple_full()
        out = paddle.concat([y3, y2, y1])
        return self.relu(out)

    def forward(self, x):
        outs = []
        for fn in [self.concat_case_1, self.concat_case_2, self.concat_case_3]:
            # to tigger duplicate subgraph and cache them.
            for i in range(3):
                outs.append(self.relu(fn()))
        outs.append(self.relu(x))
        return outs


class TestLayer(unittest.TestCase):
    def setUp(self):
        self.inputs = (paddle.rand(shape=[12], dtype=paddle.float32),)
        self.net = LayerCase()

    def eval(self, net, to_static, with_prim=False, with_cinn=False):
        if to_static:
            paddle.set_flags({'FLAGS_prim_all': with_prim})
            if with_cinn:
                build_strategy = paddle.static.BuildStrategy()
                build_strategy.build_cinn_pass = True
                net = paddle.jit.to_static(
                    net, build_strategy=build_strategy, full_graph=True
                )
            else:
                net = paddle.jit.to_static(net, full_graph=True)
        paddle.seed(123)
        net.eval()
        outs = net(*self.inputs)
        return outs

    def check_with_flag(self, cache_size):
        st_out = self.eval(self.net, to_static=True)
        cinn_out = self.eval(
            self.net, to_static=True, with_prim=True, with_cinn=True
        )
        for st, cinn in zip(
            paddle.utils.flatten(st_out), paddle.utils.flatten(cinn_out)
        ):
            np.testing.assert_allclose(st.numpy(), cinn.numpy(), atol=1e-6)

        # Check cache size
        np.testing.assert_equal(
            core.pir.cinn_compilation_cache_size(), cache_size
        )

    def test_ast_prim_cinn(self):
        # NOTE(Aurelius84): Deny relu to split fused subgraph.
        paddle.set_flags(
            {
                "FLAGS_deny_cinn_ops": "relu",
                "FLAGS_prim_forward_blacklist": "pd_op.relu",
            }
        )
        self.check_with_flag(cache_size=3)

    def test_ast_prim_cinn_disable_cache(self):
        core.pir.clear_cinn_compilation_cache()
        # NOTE(Aurelius84): Deny relu to split fused subgraph.
        paddle.set_flags(
            {
                "FLAGS_deny_cinn_ops": "relu",
                "FLAGS_prim_forward_blacklist": "pd_op.relu",
                "FLAGS_enable_cinn_compile_cache": False,
            }
        )
        # if disable cinn_compile_caceh, each subgraph will be considered as unqiue.
        self.check_with_flag(cache_size=9)


if __name__ == '__main__':
    unittest.main()
