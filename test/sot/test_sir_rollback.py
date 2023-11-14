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

from __future__ import annotations

import inspect
import operator
import unittest

from test_case_base import TestCaseBase

import paddle
from paddle.jit.sot.opcode_translator.executor.function_graph import (
    FunctionGraph,
)
from paddle.jit.sot.opcode_translator.executor.tracker import (
    DanglingTracker,
    LocalTracker,
)
from paddle.jit.sot.opcode_translator.executor.variables import (
    BuiltinVariable,
    VariableFactory,
)


def compute(x, y):
    ret = BuiltinVariable(operator.add, x.graph, DanglingTracker())(x, y)
    return BuiltinVariable(operator.mul, x.graph, DanglingTracker())(ret, x)


def try_add(x, y):
    return BuiltinVariable(operator.add, x.graph, DanglingTracker())(x, y)


class TestRollback(TestCaseBase):
    def test_rollback(self):
        frame = inspect.currentframe()
        graph = FunctionGraph(frame)
        a = paddle.to_tensor(1.0)
        b = paddle.to_tensor(2.0)
        a = VariableFactory().from_value(a, graph, LocalTracker("a"))
        b = VariableFactory().from_value(b, graph, LocalTracker("b"))
        out = compute(a, b)
        original_length = len(graph.sir_ctx.TOS.statements)
        memo = graph.save_memo()
        try_add(out, out)

        assert len(graph.sir_ctx.TOS.statements) != len(
            memo.stmt_ir.statements
        ), "After add, we must statement IR."
        graph.restore_memo(memo)

        assert len(graph.sir_ctx.TOS.statements) == original_length


def fn_with_side_effects_inner(x, y):
    x[0] += 10
    x[1] += 20
    x[2] -= 10
    print(y)  # print will cause breakgraph


def fn_with_side_effects(x, y):
    x[0] += 1
    fn_with_side_effects_inner(x, y)
    return x[0] + y


class TestSideEffectRollback(TestCaseBase):
    def test_side_effect_rollback(self):
        self.assert_results_with_side_effects(
            fn_with_side_effects, [1, 2, 3], paddle.to_tensor(42)
        )


if __name__ == "__main__":
    unittest.main()
