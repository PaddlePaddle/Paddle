#!/usr/bin/env python3

# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
from math import isclose

import cinn
from cinn import ir


class TestPackedFunc(unittest.TestCase):
    def setUp(self):
        pass

    def test_lambda(self):
        add3 = ir.register_packed_func("test_packed_func_add3")(
            lambda x, y, z: x + y + z
        )
        self.assertEqual(add3(1, 2, 3), 6)
        self.assertEqual(ir.get_global_func("test_packed_func_add3"), add3)
        self.assertTrue(isinstance(add3, ir.PackedFunc))

    def test_normal_function(self):
        @ir.register_packed_func("test_packed_func_mul")
        def mul(x, y):
            return x * y

        self.assertTrue(isclose(mul(2.3, 3.0), 6.9, abs_tol=1e-5))
        self.assertEqual(mul(4, 5), 20)

    def test_callable_object(self):
        class Accumulator:
            def __init__(self, init):
                self.init = init

            def __call__(self, *args):
                r = cinn.CINNValue(self.init)
                for arg in args:
                    r = r + arg
                return r

        accumulate = ir.register_packed_func("accumulate_float")(
            Accumulator(1.0)
        )
        self.assertTrue(isclose(accumulate(1.0, 2.0, 3.0, 4.0), 11.0))

    def test_cxx_register(self):
        add_int = ir.Registry.get("test_add_int64")
        self.assertEqual(add_int(2, 3), 5)

        add_expr = ir.Registry.get("test_add_expr")
        x = ir.Expr(1)
        y = ir.Expr(2)
        z = x + y
        r = add_expr(x, y)
        self.assertEqual(r.node_type(), z.node_type())

        mul_float = ir.Registry.get("test_mul_float")
        self.assertTrue(isclose(mul_float(2.4, 2.5), 6.0, abs_tol=1e-5))


if __name__ == "__main__":
    unittest.main()
