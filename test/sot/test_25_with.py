# Copyright (c) 2025 paddlepaddle Authors. All Rights Reserved.
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
)

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph
from paddle.jit.sot.utils import strict_mode_guard


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
def with_manager_normal(x):
    with Manager() as mgr:
        x *= 2
    return x


@check_no_breakgraph
def with_manager_exit_true_raise_error(x):
    with ManagerExitReturnTrue() as mgr_true:
        x *= 3
        raise ValueError("test")
        x -= 4
    return x


@check_no_breakgraph
def with_manager_exit_true_zero_division(x):
    with ManagerExitReturnTrue() as mgr_true:
        x += 5
        # TODO(DrRyanHuang): Division by zero (x / 0) will raise an InnerError.
        # In the future, the actual Exception should be propagated rather than being wrapped as InnerError.
        1 / 0  # noqa: B018
        x *= 6
    return x


@check_no_breakgraph
def with_contextmanager_flag_behavior(x):
    global TEST_WITH_STATEMENT_FLAG
    with my_context():
        if TEST_WITH_STATEMENT_FLAG:
            x /= 7
        else:
            x *= 7

    if not TEST_WITH_STATEMENT_FLAG:
        x += 8
    return x


@check_no_breakgraph
def with_manager_exit_false(x):
    try:
        with ManagerExitReturnFalse() as mgr_false:
            x *= 4
            1 / 0  # noqa: B018
    except ZeroDivisionError:
        x /= 4
    return x


# TODO(DrRyanHuang): NoGradContextManagerVariable and UserDefinedContextManagerVariable will be implemented separately in the future.
# The @strict_mode_guard decorator will be removed here to ensure that fallback is no longer permitted.
@strict_mode_guard(False)
def test_no_grad_behavior():
    x = paddle.rand([1, 2])
    p = paddle.rand([1, 2])
    p.stop_gradient = False
    x.stop_gradient = True
    with paddle.no_grad():
        y = (x * p).sum()
    y.backward()
    return x.grad, p.grad


class TestWithStatement(TestCaseBase):
    def test_manager_normal(self):
        t = paddle.to_tensor(-10.0)
        self.assert_results(with_manager_normal, t)

    def test_manager_exit_true_suppresses(self):
        t = paddle.to_tensor(-10.0)
        self.assert_results(with_manager_exit_true_raise_error, t)

    def test_manager_exit_true_zero_division(self):
        t = paddle.to_tensor(-10.0)
        self.assert_results(with_manager_exit_true_zero_division, t)

    def test_my_context_flag_behavior(self):
        t = paddle.to_tensor(-10.0)
        self.assert_results(with_contextmanager_flag_behavior, t)

    def test_with_manager_exit_false(self):
        t = paddle.to_tensor(-10.0)
        self.assert_results(with_manager_exit_false, t)

    def test_no_grad(self):
        self.assert_results(test_no_grad_behavior)


if __name__ == '__main__':
    unittest.main()
