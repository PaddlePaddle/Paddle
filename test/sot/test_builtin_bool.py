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
import operator
import unittest

from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
    test_with_faster_guard,
)

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph
from paddle.jit.sot.utils import strict_mode_guard


class TestObject:
    pass


class TestObjectWithBool:
    def __bool__(self):
        return False


class TestObjectWithLen:
    def __init__(self, list):
        self.list = list

    def __len__(self):
        return len(self.list)


class TestObjectWithBoolAndLen:
    def __init__(self, list):
        self.list = list

    def __bool__(self):
        return False

    def __len__(self):
        return len(self.list)


def call_bool_in_cond(obj):
    if obj:
        return True
    else:
        return False


def call_bool_by_bool(obj):
    return bool(obj)


def call_bool_by_operator_truth(obj):
    return operator.truth(obj)


class TestBuiltinBool(TestCaseBase):
    def test_object_disallow_breakgraph(self):
        call_bool_in_cond_no_breakgraph = check_no_breakgraph(call_bool_in_cond)
        call_bool_by_bool_no_breakgraph = check_no_breakgraph(call_bool_by_bool)
        call_bool_by_operator_truth_no_breakgraph = check_no_breakgraph(
            call_bool_by_operator_truth
        )

        with test_instruction_translator_cache_context():
            obj = TestObject()
            self.assert_results(call_bool_in_cond_no_breakgraph, obj)
            self.assert_results(call_bool_by_bool_no_breakgraph, obj)
            self.assert_results(call_bool_by_operator_truth_no_breakgraph, obj)

        with test_instruction_translator_cache_context():
            obj = TestObjectWithBool()
            self.assert_results(call_bool_in_cond_no_breakgraph, obj)
            self.assert_results(call_bool_by_bool_no_breakgraph, obj)
            self.assert_results(call_bool_by_operator_truth_no_breakgraph, obj)

        with test_instruction_translator_cache_context():
            obj = TestObjectWithBoolAndLen([1, 2, 3])
            self.assert_results(call_bool_in_cond_no_breakgraph, obj)
            self.assert_results(call_bool_by_bool_no_breakgraph, obj)
            self.assert_results(call_bool_by_operator_truth_no_breakgraph, obj)

        with test_instruction_translator_cache_context():
            obj = TestObjectWithBoolAndLen([])
            self.assert_results(call_bool_in_cond_no_breakgraph, obj)
            self.assert_results(call_bool_by_bool_no_breakgraph, obj)
            self.assert_results(call_bool_by_operator_truth_no_breakgraph, obj)

        with test_instruction_translator_cache_context():
            layer = paddle.nn.Linear(10, 1)
            self.assert_results(call_bool_in_cond_no_breakgraph, layer)
            self.assert_results(call_bool_by_bool_no_breakgraph, layer)
            self.assert_results(
                call_bool_by_operator_truth_no_breakgraph, layer
            )

    @test_with_faster_guard
    def test_object_allow_breakgraph(self):
        with test_instruction_translator_cache_context():
            obj = TestObjectWithLen([1, 2, 3])
            with strict_mode_guard(False):
                self.assert_results(call_bool_in_cond, obj)

            self.assert_results(call_bool_by_bool, obj)
            self.assert_results(call_bool_by_operator_truth, obj)

        with test_instruction_translator_cache_context():
            obj = TestObjectWithLen([])
            with strict_mode_guard(False):
                self.assert_results(call_bool_in_cond, obj)

            self.assert_results(call_bool_by_bool, obj)
            self.assert_results(call_bool_by_operator_truth, obj)


if __name__ == "__main__":
    unittest.main()
