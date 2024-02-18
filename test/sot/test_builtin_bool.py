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

from test_case_base import TestCaseBase

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


@check_no_breakgraph
def object_bool(obj):
    if obj:
        return True
    else:
        return False


@strict_mode_guard(False)
def object_bool_allow_breakgraph(obj):
    if obj:
        return True
    else:
        return False


@check_no_breakgraph
def test_bool(obj):
    return bool(obj)


@check_no_breakgraph
def test_operator_truth(obj):
    return operator.truth(obj)


def test_bool_allow_breakgraph(obj):
    return bool(obj)


def test_operator_truth_allow_breakgraph(obj):
    return operator.truth(obj)


class TestBuiltinBool(TestCaseBase):
    def test_object(self):
        object = TestObject()
        self.assert_results(object_bool, object)
        self.assert_results(object_bool, bool(object))
        self.assert_results(object_bool, operator.truth(object))
        self.assert_results(test_bool, object)
        self.assert_results(test_operator_truth, object)

    def test_object_with_bool(self):
        object = TestObjectWithBool()
        self.assert_results(object_bool, object)
        self.assert_results(object_bool, bool(object))
        self.assert_results(object_bool, operator.truth(object))
        self.assert_results(test_bool, object)
        self.assert_results(test_operator_truth, object)

    def test_object_with_len(self):
        object = TestObjectWithLen([1, 2, 3])
        self.assert_results(object_bool_allow_breakgraph, object)
        self.assert_results(object_bool_allow_breakgraph, bool(object))
        self.assert_results(
            object_bool_allow_breakgraph, operator.truth(object)
        )
        self.assert_results(test_bool_allow_breakgraph, object)
        self.assert_results(test_operator_truth_allow_breakgraph, object)

        object = TestObjectWithLen([])
        self.assert_results(object_bool_allow_breakgraph, object)
        self.assert_results(object_bool_allow_breakgraph, bool(object))
        self.assert_results(
            object_bool_allow_breakgraph, operator.truth(object)
        )
        self.assert_results(test_bool_allow_breakgraph, object)
        self.assert_results(test_operator_truth_allow_breakgraph, object)

    def test_object_with_bool_and_len(self):
        object = TestObjectWithBoolAndLen([1, 2, 3])
        self.assert_results(object_bool, object)
        self.assert_results(object_bool, bool(object))
        self.assert_results(object_bool, operator.truth(object))
        self.assert_results(test_bool, object)
        self.assert_results(test_operator_truth, object)

        object = TestObjectWithBoolAndLen([])
        self.assert_results(object_bool, object)
        self.assert_results(object_bool, bool(object))
        self.assert_results(object_bool, operator.truth(object))
        self.assert_results(test_bool, object)
        self.assert_results(test_operator_truth, object)

    def test_layer(self):
        layer = paddle.nn.Linear(10, 1)
        self.assert_results(object_bool, layer)
        self.assert_results(object_bool, bool(layer))
        self.assert_results(object_bool, operator.truth(layer))
        self.assert_results(test_bool, layer)
        self.assert_results(test_operator_truth, layer)


if __name__ == "__main__":
    unittest.main()
