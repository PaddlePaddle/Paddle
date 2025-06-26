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

import unittest
from contextlib import contextmanager

from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
)

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph


class Manager:
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc, value, traceback):
        pass


class ManagerExitReturnFalse(Manager):
    def __exit__(self, *args):
        return False


class ManagerExitReturnTrue(Manager):
    def __exit__(self, *args):
        return True


TEST_WITH_STATEMENT_FLAG = False


@contextmanager
def my_context():
    global TEST_WITH_STATEMENT_FLAG
    try:
        TEST_WITH_STATEMENT_FLAG = True
        yield
    finally:
        TEST_WITH_STATEMENT_FLAG = False


@check_no_breakgraph
def test_with_exit_true_suppresses(x):
    x += 1

    with Manager() as mgr:
        x *= 2

    with ManagerExitReturnTrue() as mgr_true:
        x *= 3
        raise ValueError("test")
        x -= 4

    with ManagerExitReturnTrue() as mgr_true:
        x += 5
        # TODO(DrRyanHuang): Division by zero (x / 0) will raise an InnerError.
        # In the future, the actual Exception should be propagated rather than being wrapped as InnerError.
        1 / 0  # noqa: B018
        x *= 6

    global TEST_WITH_STATEMENT_FLAG

    with my_context() as e:
        if TEST_WITH_STATEMENT_FLAG:
            x /= 7
        else:
            x *= 7

    if not TEST_WITH_STATEMENT_FLAG:
        x += 8
    else:
        x -= 8

    x /= 9
    return x


@check_no_breakgraph
def test_with_exit_false_propagates(x):
    x += 3
    try:
        with ManagerExitReturnFalse() as mgr_false:
            x *= 4
            1 / 0  # noqa: B018
    except ZeroDivisionError:
        x /= 4
    return x


class TestWithStatement(TestCaseBase):
    def test_with(self):
        t = paddle.to_tensor(-10.0)
        self.assert_results(test_with_exit_true_suppresses, t)
        self.assert_results(test_with_exit_false_propagates, t)

    def test_guard_run(self):
        x = paddle.to_tensor([-4.0])
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            self.assert_results(test_with_exit_true_suppresses, x)
            self.assert_results(test_with_exit_true_suppresses, x)
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(test_with_exit_false_propagates, x)
            self.assert_results(test_with_exit_false_propagates, x)
            self.assertEqual(ctx.translate_count, 2)

    def test_no_grad(self):
        x = paddle.rand([1, 2])
        layer1 = paddle.nn.Linear(2, 2)
        y = layer1(x).sum()
        self.assertTrue(layer1.weight.grad is None)
        y.backward()
        self.assertFalse(layer1.weight.grad is None)

        layer2 = paddle.nn.Linear(2, 2)
        with paddle.no_grad():
            y = layer2(x).sum()
        self.assertTrue(layer2.weight.grad is None)
        y.backward()
        self.assertTrue(layer2.weight.grad is None)


if __name__ == '__main__':
    unittest.main()
