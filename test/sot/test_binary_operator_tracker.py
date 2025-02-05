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

# GET_ITER (new)
# FOR_ITER (new)

from __future__ import annotations

import os

os.environ['MIN_GRAPH_SIZE'] = '-1'
import operator
import unittest

from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
)

import paddle
from paddle.jit.sot.opcode_translator.executor.dispatcher import (
    Dispatcher,
    Parameter,
    Pattern,
)
from paddle.jit.sot.opcode_translator.executor.tracker import (
    BinaryOperatorTracker,
)
from paddle.jit.sot.opcode_translator.executor.variables import ConstantVariable


def register_dispatch(fn, parameters, handler):
    """
    Registering function signature.

    Args:
        fn: The function to be registered.
        parameters: The parameters of the function to be registered.
        handler: The handler function.
    """
    _parameters = tuple(
        (
            Parameter.from_str(parameter)
            if isinstance(parameter, str)
            else parameter
        )
        for parameter in parameters
    )
    if fn not in Dispatcher.handlers:
        Dispatcher.handlers[fn] = []
    Dispatcher.handlers[fn].insert(0, (Pattern(*_parameters), handler))


def operator_gt_dis_func(left, right):
    return ConstantVariable(
        operator.gt(left.get_py_value(), right.get_py_value()),
        graph=left.graph,
        tracker=BinaryOperatorTracker("COMPARE_OP", [left, right], 4),
    )


def operator_add_dis_func(left, right):
    return ConstantVariable(
        operator.gt(left.get_py_value(), right.get_py_value()),
        graph=left.graph,
        tracker=BinaryOperatorTracker("BINARY_ADD", [left, right]),
    )


class TestBinaryOperatorTracker(TestCaseBase):
    def test_case_compare_op(self):
        def func(x, y):
            if x > 0:
                return y + 1
            return y + 2

        register_dispatch(
            operator.gt,
            ("ConstantVariable", "ConstantVariable"),
            operator_gt_dis_func,
        )

        y = paddle.randn((2, 2))
        with test_instruction_translator_cache_context() as ctx:
            self.assert_results(func, 12, y)
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(func, 10, y)
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(func, -12, y)
            self.assertEqual(ctx.translate_count, 2)

    def test_case_compare_op_2(self):
        def func(x, y):
            if x + 2 > 0:
                return y + 1
            return y + 2

        register_dispatch(
            operator.gt,
            ("ConstantVariable", "ConstantVariable"),
            operator_gt_dis_func,
        )

        register_dispatch(
            operator.add,
            ("ConstantVariable", "ConstantVariable"),
            operator_add_dis_func,
        )

        y = paddle.randn((2, 2))
        with test_instruction_translator_cache_context() as ctx:
            self.assert_results(func, 12, y)
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(func, 10, y)
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(func, -12, y)
            self.assertEqual(ctx.translate_count, 2)


if __name__ == "__main__":
    unittest.main()
