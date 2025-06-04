# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import random
import unittest

from test_case_base import TestCaseBase

import paddle
from paddle.jit.sot import symbolic_translate


def fn_randint(x):
    x = x + 1
    x = x + random.randint(0, 100)
    x = x + 2
    return x


def fn_random(x):
    x = x + 3
    x = x + random.random()
    x = x + 4
    return x


class TestRandom(TestCaseBase):
    def test_random_randint(self):
        x = paddle.to_tensor(2024)

        random.seed(2025)
        sym_output = symbolic_translate(fn_randint)(x)
        random.seed(2025)
        paddle_output = fn_randint(x)

        self.assertEqual(sym_output, paddle_output)

    def test_random_random(self):
        x = paddle.to_tensor(2025)

        random.seed(2025)
        sym_output = symbolic_translate(fn_random)(x)
        random.seed(2025)
        paddle_output = fn_random(x)

        self.assertEqual(sym_output, paddle_output)


if __name__ == "__main__":
    unittest.main()
