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

import unittest

from test_case_base import test_instruction_translator_cache_context

import paddle
from paddle.jit.sot import (
    psdb,
    symbolic_translate,
)
from paddle.jit.sot.utils.envs import strict_mode_guard
from paddle.jit.sot.utils.exceptions import InnerError


def assert_true_case(input: bool):
    psdb.assert_true(input)


def breakgraph_case(x):
    x = x + 1
    psdb.breakgraph()
    x = x + 1
    return x


def fallback_not_recursive_inner(x):
    x = x + 1
    x = x + 1
    return x


def fallback_not_recursive_case(x):
    psdb.fallback(recursive=False)
    x = fallback_not_recursive_inner(x)
    return x


def fallback_recursive_case(x):
    psdb.fallback(recursive=True)
    x = fallback_not_recursive_inner(x)
    return x


@psdb.check_no_breakgraph
def check_no_breakgraph_case(x):
    x = x + 1
    psdb.breakgraph()
    x = x + 1
    return x


@psdb.check_no_fallback
def check_no_fallback_case(x):
    x = x + 1
    psdb.fallback(recursive=False)
    x = x + 1
    return x


class TestPsdb(unittest.TestCase):
    def test_assert_true(self):
        # Test with True
        symbolic_translate(assert_true_case)(True)

        # Test with False
        with self.assertRaises(InnerError):
            symbolic_translate(assert_true_case)(False)

    def test_breakgraph(self):
        x = paddle.to_tensor([1.0])
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            symbolic_translate(breakgraph_case)(x)
            self.assertEqual(ctx.translate_count, 2)

    @strict_mode_guard(False)
    def test_fallback_not_recursive(self):
        x = paddle.to_tensor([1.0])
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            symbolic_translate(fallback_not_recursive_case)(x)
            self.assertEqual(ctx.translate_count, 2)

    @strict_mode_guard(False)
    def test_fallback_recursive(self):
        x = paddle.to_tensor([1.0])
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            symbolic_translate(fallback_recursive_case)(x)
            self.assertEqual(ctx.translate_count, 1)

    def test_check_no_breakgraph(self):
        x = paddle.to_tensor([1.0])
        with self.assertRaises(InnerError):
            symbolic_translate(check_no_breakgraph_case)(x)

    @strict_mode_guard(False)
    def test_check_no_fallback(self):
        x = paddle.to_tensor([1.0])
        with self.assertRaises(InnerError):
            symbolic_translate(check_no_fallback_case)(x)


if __name__ == "__main__":
    unittest.main()
