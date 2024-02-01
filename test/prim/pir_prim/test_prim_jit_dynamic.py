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


def rms_norm1(hidden_states, weight):
    # From llama2, reduce dim is not equal to dynamic shape dim
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = paddle.rsqrt(variance + 1e-5) * hidden_states
    return hidden_states * weight


def rms_norm2(hidden_states, weight):
    # reduce dim is not equal to dynamic shape dim
    variance = hidden_states.pow(2).mean((0, 1), keepdim=True)
    hidden_states = paddle.rsqrt(variance + 1e-5) * hidden_states
    return hidden_states * weight


def log_softmax_net(x):
    return F.log_softmax(x)


def any_net(x):
    return paddle.any(x)


class TestPrimMode1(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [1, 300, 4096]
        self.shape_y = [4096]
        self.x = np.random.random(self.shape_x).astype("float32")
        self.y = np.random.random(self.shape_y).astype("float32")
        self.net = rms_norm1

    def base_net(self, flag=None):
        if flag == "prim":
            core._set_prim_all_enabled(True)
        x = paddle.to_tensor(self.x)
        y = paddle.to_tensor(self.y)
        fn = apply_to_static(
            self.net,
            use_cinn=False,
            input_spec=[
                InputSpec(shape=[None, None, 4096], dtype='float32'),
                InputSpec(shape=[4096], dtype='float32'),
            ],
        )
        res = fn(x, y)
        ops = [
            op.name()
            for op in fn.program_cache.last()[-1][-1]
            .infer_program.program.global_block()
            .ops
        ]

        if flag == "prim":
            assert "pd_op.mean" not in ops
            core._set_prim_all_enabled(False)
        else:
            assert "pd_op.mean" in ops
        return res

    def test_prim_all_dynamic(self):
        res_ref = self.base_net()
        res = self.base_net("prim")
        for ref, actual in zip(res_ref, res):
            np.testing.assert_allclose(ref, actual, rtol=1e-6)


class TestPrimMode2(TestPrimMode1):
    def setUp(self):
        np.random.seed(2023)
        self.shape_x = [1, 300, 4096]
        self.shape_y = [4096]
        self.x = np.random.random(self.shape_x).astype("float32")
        self.y = np.random.random(self.shape_y).astype("float32")
        self.net = rms_norm2


class TestPrimOne(unittest.TestCase):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.shape_x = [1, 300, 4096]
        self.x = np.random.random(self.shape_x).astype(self.dtype)
        self.net = log_softmax_net
        self.necessary_ops = "pd_op.log_softmax"

    def base_net(self, flag=None):
        if flag == "prim":
            core._set_prim_all_enabled(True)
        x = paddle.to_tensor(self.x)
        fn = apply_to_static(
            self.net,
            use_cinn=False,
            input_spec=[
                InputSpec(shape=[None, None, 4096], dtype=self.dtype),
            ],
        )
        res = fn(x)
        ops = [
            op.name()
            for op in fn.program_cache.last()[-1][-1]
            .infer_program.program.global_block()
            .ops
        ]

        if flag == "prim":
            assert self.necessary_ops not in ops
            core._set_prim_all_enabled(False)
        else:
            assert self.necessary_ops in ops
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


if __name__ == "__main__":
    unittest.main()
