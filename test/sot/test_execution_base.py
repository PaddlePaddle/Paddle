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

from test_case_base import TestCaseBase

import paddle
from paddle.framework import use_pir_api
from paddle.jit.sot import symbolic_translate
from paddle.static import BuildStrategy


def func(x, y):
    ret = 2 * x
    ret = paddle.nn.functional.relu(ret)
    ret = ret + y
    return ret


def simple(x):
    ret = 2 * x
    return ret


class TestExecutionBase(TestCaseBase):
    def test_simple(self):
        x = paddle.to_tensor([1.0])
        y = paddle.to_tensor([2.0])
        self.assert_results(simple, x)
        self.assert_results(simple, y)


def foo(x):
    out = x + 1
    out = out * 2
    out = paddle.nn.functional.relu(out)
    return out


class TestBackend(TestCaseBase):
    def test_backend(self):
        x = paddle.randn([2, 3])
        dy_out = foo(x)
        # TODO(SigureMo): Find a better way to test the CINN backend.
        if not paddle.is_compiled_with_cinn() and use_pir_api():
            return
        sot_out = symbolic_translate(
            foo, build_strategy=BuildStrategy(), backend='CINN'
        )(x)
        self.assert_nest_match(dy_out, sot_out)


if __name__ == "__main__":
    unittest.main()
