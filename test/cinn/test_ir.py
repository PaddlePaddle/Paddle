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

from paddle.cinn.ir import Expr, Var
from paddle.cinn.optim import simplify


class TestIR(unittest.TestCase):
    def test_pod(self):
        one = Expr(1)
        self.assertEqual(str(simplify(one + one)), "2")
        self.assertEqual(str(simplify(one * Expr(0))), "0")

    def test_expr(self):
        a = Var("A")
        b = Var("B")

        expr = 1 + b
        print(expr)

        expr = b + 1
        print(expr)

        self.assertEqual(str(simplify(b * 0)), "0")
        print(expr)
        print(simplify(expr))


if __name__ == "__main__":
    unittest.main()
