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

import unittest

from test_case_base import TestCaseBase

import paddle
from paddle.jit.sot.utils import strict_mode_guard


def foo(x: int, y: paddle.Tensor):
    return x + y


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(foo, 1, paddle.to_tensor(2))


def numpy_add(x, y):
    out = paddle.to_tensor(x.numpy() + y.numpy())
    return out


class TestNumpyAdd(TestCaseBase):
    @strict_mode_guard(False)
    def test_numpy_add(self):
        x = paddle.to_tensor([2])
        y = paddle.to_tensor([3])
        self.assert_results(numpy_add, x, y)


if __name__ == "__main__":
    unittest.main()


# Instructions:
# LOAD_FAST
# BINARY_ADD
# RETURN_VALUE

# Variables:
# ConstantVariable
# TensorVariable
