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

from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
)

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph


class Manager:
    def __init__(self, x):
        x /= 1

    def __enter__(self):
        return self

    def __exit__(self, exc, value, traceback):
        pass

    def add1(self, x):
        x += 1
        return x

    def sub2(self, x):
        x -= 2
        return x

    def mul3(self, x):
        x *= 3
        return x

    def div4(self, x):
        x /= 4
        return x


class ManagerReturnFalse(Manager):
    def __exit__(self, *args):
        return False


class ManagerReturnTrue(Manager):
    def __exit__(self, *args):
        return True


@check_no_breakgraph
def run_with_stmt(x):
    x += 1

    with Manager(x) as mgr:
        x *= 2
        mgr.add1(x)

    with ManagerReturnTrue(x) as mgr_true:
        x *= 3
        mgr_true.div4(x)
        raise ValueError("test")
        x -= 4
        mgr_true.sub2(x)

    with ManagerReturnTrue(x) as mgr_true:
        x /= 5
        mgr_true.add1(x)
        x / 0
        x -= 6
        mgr_true.mul3(x)

    x -= 7
    mgr.div4(x)
    return x


@check_no_breakgraph
def run_with_stmt_with_error(x):
    x += 3
    try:
        with ManagerReturnFalse(x) as mgr_false:
            x *= 4
            mgr_false.add1(x)
            x / 0
    except ZeroDivisionError:
        x /= 4
        mgr_false.sub2(x)
    return x


class TestWithStatement(TestCaseBase):
    def test_with(self):
        t = paddle.to_tensor(-10.0)
        self.assert_results(run_with_stmt, t)

    def test_with_with_try_except(self):
        t = paddle.to_tensor(123.0)
        self.assert_results(run_with_stmt, t)

    def test_guard_run(self):
        x = paddle.to_tensor([-4.0])
        with test_instruction_translator_cache_context() as ctx:
            self.assertEqual(ctx.translate_count, 0)
            self.assert_results(run_with_stmt, x)
            self.assert_results(run_with_stmt, x)
            self.assert_results(run_with_stmt, x)
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(run_with_stmt_with_error, x)
            self.assert_results(run_with_stmt_with_error, x)
            self.assert_results(run_with_stmt_with_error, x)
            self.assertEqual(ctx.translate_count, 2)


if __name__ == '__main__':
    unittest.main()
