# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.nn.functional as F
from paddle.framework import core
from paddle.static import InputSpec


def apply_to_static(net, use_cinn, input_spec=None):
    build_strategy = paddle.static.BuildStrategy()
    build_strategy.build_cinn_pass = use_cinn
    return paddle.jit.to_static(
        net,
        input_spec=input_spec,
        build_strategy=build_strategy,
        full_graph=True,
    )


def log_softmax_net(x):
    return F.log_softmax(x)


def any_net(x):
    return paddle.any(x)


def embedding_net(x):
    w = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
            [12, 13, 14],
            [15, 16, 17],
            [18, 19, 20],
            [21, 22, 23],
            [24, 25, 26],
            [27, 28, 29],
        ],
        dtype=np.float32,
    )
    w = paddle.to_tensor(w)
    return F.embedding(x, w, padding_idx=1)


class TestPrimOne(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.shape_x = [1, 300, 4096]
        self.x = np.random.random(self.shape_x).astype(self.dtype)
        self.net = log_softmax_net
        self.necessary_ops = "pd_op.log_softmax"
        self.enable_cinn = False

    def base_net(self, flag=None):
        if flag == "prim":
            core._set_prim_all_enabled(True)
        x = paddle.to_tensor(self.x)
        if flag == "prim":
            fn = apply_to_static(
                self.net,
                use_cinn=self.enable_cinn,
                input_spec=[
                    InputSpec(shape=[None, None, 4096], dtype='float32'),
                ],
            )
            fn.eval()
        else:
            fn = self.net
        res = fn(x)

        if flag == "prim":
            ops = [
                op.name()
                for op in fn.program_cache.last()[-1][-1]
                .infer_program.program.global_block()
                .ops
            ]
            assert self.necessary_ops not in ops
            core._set_prim_all_enabled(False)
        return res

    def test_prim_all_dynamic(self):
        res_ref = self.base_net()
        res = self.base_net("prim")
        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(ref, actual, rtol=1e-6)


class TestPrimOne2(TestPrimOne):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "bool"
        self.shape_x = [1, 300, 4096]
        self.x = np.random.random(self.shape_x).astype(self.dtype)
        self.net = any_net
        self.necessary_ops = "pd_op.any"
        self.enable_cinn = False


# class TestEmbeddingPrimOne3(TestPrimOne):
#     def setUp(self):
#         np.random.seed(2023)
#         self.dtype = "int"
#         self.shape_x = [1, 300, 4096]
#         self.x = np.random.randint(0, 10, size=self.shape_x)
#         self.net = embedding_net
#         self.necessary_ops = "pd_op.embedding"
#         self.enable_cinn = False


if __name__ == "__main__":
    unittest.main()
