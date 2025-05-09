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
from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
)

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph
from paddle.jit.sot.utils import strict_mode_guard


def numpy_add(x, y):
    out = paddle.to_tensor(x.numpy() + y.numpy())
    return out


def tensor_add_numpy(x, y):
    ret = x + y
    return ret


def large_numpy_array_to_tensor(x):
    return paddle.to_tensor(x)


def normal_numpy_array_to_tensor(x):
    return paddle.to_tensor(x)


@check_no_breakgraph
def numpy_api_with_number_calculation(t):
    a = np.log(2)
    b = np.exp(3)
    c = np.sqrt(4)
    d = np.ceil(5.1)
    e = np.add(1, 2)
    f = a + 1
    g = 1 - b
    h = c * 2
    i = int(a)
    j = float(b)
    k = c.item()
    l = t + d
    return a, b, c, d, e, f, g, h, i, j, k, l


@check_no_breakgraph
def numpy_bool(x: np.number):
    return bool(x == 1)


class TestNumPy(TestCaseBase):
    @strict_mode_guard(False)
    def test_numpy_add(self):
        x = paddle.to_tensor([2])
        y = paddle.to_tensor([3])
        self.assert_results(numpy_add, x, y)

    def test_tensor_add_numpy_number(self):
        x = paddle.to_tensor([1.0])
        y = np.int64(2)
        self.assert_results(tensor_add_numpy, x, y)
        self.assert_results(tensor_add_numpy, y, x)

    @strict_mode_guard(False)
    def test_tensor_add_numpy_array(self):
        x = paddle.to_tensor([1.0])
        y = np.array(2.0)
        self.assert_results(tensor_add_numpy, x, y)
        self.assert_results(tensor_add_numpy, y, x)

    def test_large_numpy_array_to_tensor(self):
        # size should be larger than 1024*1024, because we throw an exception
        # when the size is larger than 1024*1024 in assign API (to_tensor static branch)
        x = np.random.rand(1024, 1024, 2).astype(np.float32)
        self.assert_results(large_numpy_array_to_tensor, x)

    def test_numpy_array_guard(self):
        x = np.array([1.0, 2.0])
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            self.assert_results(normal_numpy_array_to_tensor, x)
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(normal_numpy_array_to_tensor, x)
            self.assertEqual(ctx.translate_count, 1)

    def test_numpy_api_with_number_calculation(self):
        t = paddle.to_tensor([1.0])
        self.assert_results(numpy_api_with_number_calculation, t)

    def test_numpy_bool(self):
        x = np.float32(1.0)
        self.assert_results(numpy_bool, x)


if __name__ == "__main__":
    unittest.main()
