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
import sys
import unittest

import paddle
from paddle.jit.sot.opcode_translator.instruction_utils import (
    analysis_used_names,
    calc_offset_from_bytecode_offset,
    get_instructions,
)
from paddle.jit.sot.opcode_translator.instruction_utils.opcode_info import (
    PYOPCODE_CACHE_SIZE,
)


def assert_inputs_equals(instruction_offset: int, expected_inputs: set[str]):
    current_frame = inspect.currentframe()
    assert current_frame is not None
    test_frame = current_frame.f_back
    assert test_frame is not None

    instructions = get_instructions(test_frame.f_code)
    current_offset = test_frame.f_lasti
    if sys.version_info >= (3, 13):
        current_offset += PYOPCODE_CACHE_SIZE.get("CALL") * 2
    current_instr_idx = calc_offset_from_bytecode_offset(
        current_offset + 2, instructions
    )
    reads, writes = analysis_used_names(
        instructions, current_instr_idx + instruction_offset
    )
    assert (
        set(reads) == expected_inputs
    ), f"actual_inputs: {reads}, expected_inputs: {expected_inputs}"


def case1(x):
    m = x + 1
    n = x + 2
    assert_inputs_equals(0, {"x", "n"})
    y = x + 2
    assert_inputs_equals(0, {"n"})
    return n


def case2(x):
    x = x + 1
    assert_inputs_equals(0, {"x"})
    y = x + 3
    z = x + y
    assert_inputs_equals(0, {"x"})
    x += 1
    m = x + 1
    n = x + m
    assert_inputs_equals(0, set())
    return 1


def case3(x):
    y = x + 1

    assert_inputs_equals(0, {"x"})
    if x:
        z = 1
    else:
        z = 2
    return z


def case4(x):
    y = x + 1

    assert_inputs_equals(0, {"x", "y"})
    if x:
        z = y
    else:
        z = x
    return z


def case5(x):
    y = x + 1
    z = x + 2

    assert_inputs_equals(0, {"z"})
    if z:
        a = 1
    else:
        b = 2
    return z


def case6(x):
    y = x + 1
    z = x + 2

    assert_inputs_equals(0, {"a", "z"})
    if z:
        a = 1
    else:
        a += 1
    return z


def case7(x):
    y = x + 1
    z = x + 2

    assert_inputs_equals(0, {"a", "z"})
    if not z:
        a += 1  # noqa: F821
    else:
        a = 1
    return z


def breakgraph_api(x):
    return x


def normal_api(x):
    return x


def case8(x):
    x = normal_api(x)
    assert_inputs_equals(0, {"x"})
    for i in range(10):
        x += 1
        if i > 5:
            continue
            x += 10086
        x += i
    return x


# NOTE(SigureMo): The offset should be between index of CALL instruction of assert_inputs_equals
# and the index of the CALL instruction of breakgraph_api
case9_offset = -7
case9_offset = -9 if sys.version_info >= (3, 11) else case9_offset
case9_offset = -6 if sys.version_info >= (3, 12) else case9_offset


def case9(x):
    x = breakgraph_api(x)
    assert_inputs_equals(
        case9_offset, set()
    )  # analysis when call breakgraph api (CALL_FUNCTION)
    for i in range(10):
        x += 1
        if i > 5:
            continue
            x += 10086
        x += i
    return x


def case10(x):
    assert_inputs_equals(0, {"x", "y"})
    # if x == 0, y will be read before assignment
    for i in range(x):
        y = i
        z = y

    return y + 1


def case11(x):
    y = x + 1
    z = x + 2

    assert_inputs_equals(0, {"a", "y", "z"})
    if z:
        if not y:
            a += 1  # noqa: F821
        else:
            a = 2
    else:
        if y:
            a = 1
        else:
            a += 1
    return z


def case12(x):
    y = x + 1
    z = x + 2

    assert_inputs_equals(0, {"a", "y", "z"})
    if z:
        if y:
            a = 2
        else:
            a += 2
    else:
        if y:
            a += 1
        else:
            a = 1
    return z


class TestAnalysisInputs(unittest.TestCase):
    def test_case1(self):
        case1(paddle.to_tensor([1]))

    def test_case2(self):
        case2(paddle.to_tensor([2]))

    def test_case3(self):
        case3(paddle.to_tensor([3]))

    def test_case4(self):
        case4(paddle.to_tensor([4]))

    def test_case5(self):
        case5(paddle.to_tensor([5]))

    def test_case6(self):
        case6(paddle.to_tensor([6]))

    def test_case7(self):
        case7(paddle.to_tensor([7]))

    def test_case8(self):
        case8(paddle.to_tensor([8]))

    def test_case9(self):
        case9(paddle.to_tensor([9]))

    def test_case10(self):
        case10(paddle.to_tensor([10]))

    def test_case11(self):
        case11(paddle.to_tensor([11]))

    def test_case12(self):
        case12(paddle.to_tensor([12]))


if __name__ == "__main__":
    unittest.main()
