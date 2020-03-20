#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
from decorator_helper import prog_scope
import paddle.fluid as fluid
import numpy as np


class TestMathOpPatchesVarBase(unittest.TestCase):
    def setUp(self):
        self.shape = [10, 10]
        self.dtype = np.float32

    def test_add(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            res = a + b
            self.assertTrue(np.array_equal(res.numpy(), a_np + b_np))

    def test_sub(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            res = a - b
            self.assertTrue(np.array_equal(res.numpy(), a_np - b_np))

    def test_mul(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            res = a * b
            self.assertTrue(np.array_equal(res.numpy(), a_np * b_np))

    def test_div(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            res = a / b
            self.assertTrue(np.array_equal(res.numpy(), a_np / b_np))

    def test_add_scalar(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = 0.1
            res = a + b
            self.assertTrue(np.array_equal(res.numpy(), a_np + b))

    def test_add_scalar_reverse(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = 0.1
            res = b + a
            self.assertTrue(np.array_equal(res.numpy(), b + a_np))

    def test_sub_scalar(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = 0.1
            res = a - b
            self.assertTrue(np.array_equal(res.numpy(), a_np - b))

    def test_sub_scalar_reverse(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = 0.1
            res = b - a
            self.assertTrue(np.array_equal(res.numpy(), b - a_np))

    def test_mul_scalar(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = 0.1
            res = a * b
            self.assertTrue(np.array_equal(res.numpy(), a_np * b))

    # div_scalar, not equal
    def test_div_scalar(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = 0.1
            res = a / b
            self.assertTrue(np.allclose(res.numpy(), a_np / b))

    # pow of float type, not equal
    def test_pow(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            res = a**b
            self.assertTrue(np.allclose(res.numpy(), a_np**b_np))

    def test_floor_div(self):
        a_np = np.random.randint(1, 100, size=self.shape)
        b_np = np.random.randint(1, 100, size=self.shape)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            res = a // b
            self.assertTrue(np.array_equal(res.numpy(), a_np // b_np))

    def test_mod(self):
        a_np = np.random.randint(1, 100, size=self.shape)
        b_np = np.random.randint(1, 100, size=self.shape)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            res = a % b
            self.assertTrue(np.array_equal(res.numpy(), a_np % b_np))

    # for logical compare
    def test_equal(self):
        a_np = np.asarray([1, 2, 3, 4, 5])
        b_np = np.asarray([1, 2, 3, 4, 5])
        c_np = np.asarray([1, 2, 2, 4, 5])
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            c = fluid.dygraph.to_variable(c_np)
            res1 = (a == b)
            res2 = (a == c)
            self.assertTrue(np.array_equal(res1.numpy(), a_np == b_np))
            self.assertTrue(np.array_equal(res2.numpy(), a_np == c_np))

    def test_not_equal(self):
        a_np = np.asarray([1, 2, 3, 4, 5])
        b_np = np.asarray([1, 2, 3, 4, 5])
        c_np = np.asarray([1, 2, 2, 4, 5])
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            c = fluid.dygraph.to_variable(c_np)
            res1 = (a != b)
            res2 = (a != c)
            self.assertTrue(np.array_equal(res1.numpy(), a_np != b_np))
            self.assertTrue(np.array_equal(res2.numpy(), a_np != c_np))

    def test_less_than(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            res = (a < b)
            self.assertTrue(np.array_equal(res.numpy(), a_np < b_np))

    def test_less_equal(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            res = (a <= b)
            self.assertTrue(np.array_equal(res.numpy(), a_np <= b_np))

    def test_greater_than(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            res = (a > b)
            self.assertTrue(np.array_equal(res.numpy(), a_np > b_np))

    def test_greater_equal(self):
        a_np = np.random.random(self.shape).astype(self.dtype)
        b_np = np.random.random(self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            b = fluid.dygraph.to_variable(b_np)
            res = (a >= b)
            self.assertTrue(np.array_equal(res.numpy(), a_np >= b_np))

    def test_neg(self):
        a_np = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        with fluid.dygraph.guard():
            a = fluid.dygraph.to_variable(a_np)
            res = -a
            self.assertTrue(np.array_equal(res.numpy(), -a_np))


if __name__ == '__main__':
    unittest.main()
