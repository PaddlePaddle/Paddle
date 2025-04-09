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

import functools
import operator
import unittest

from test_case_base import TestCaseBase

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph
from paddle.jit.sot.utils.exceptions import InnerError


@functools.lru_cache
def fn_with_cache(x):
    return x + 1


@check_no_breakgraph
def fn_with_lru_cache(x: paddle.Tensor):
    a1 = fn_with_cache(1)
    b1 = fn_with_cache(x)
    b2 = fn_with_cache(x)
    a2 = fn_with_cache(1)
    c1 = fn_with_cache(2)
    c2 = fn_with_cache(2)
    return a1, a2, b1, b2, c1, c2


@check_no_breakgraph
def try_reduce(fn, var, init=None):
    if init:
        ans = functools.reduce(fn, var, init)
    else:
        ans = functools.reduce(fn, var)
    return ans


@check_no_breakgraph
def try_reduce_iter(fn, var, init=None):
    ans = functools.reduce(fn, iter(var))
    return ans


@check_no_breakgraph
def try_reduce_iter_failed(fn, var):
    it = iter(var)
    for _ in it:
        pass
    ans = functools.reduce(fn, it)
    return ans


class TestFunctools(TestCaseBase):
    def test_lru_cache(self):
        x = paddle.rand([2, 3, 4])
        self.assert_results(fn_with_lru_cache, x)

    def test_reduce_list(self):
        self.assert_results(try_reduce, lambda acc, x: acc + x, [1, 3, 4])

    def test_reduce_dict(self):
        d1 = {"a": 1, "b": 2}
        d2 = {"a": 3, "c": 4}
        self.assert_results(try_reduce, lambda a, b: {**a, **b}, [d1, d2])

    def test_reduce_tuple(self):
        self.assert_results(try_reduce, lambda acc, x: acc * x, (1, 3, 4))

    def test_reduce_iter(self):
        f = lambda acc, x: acc + x
        l = [1, 2, 3]
        self.assert_results(try_reduce_iter, f, l)
        self.assert_exceptions(
            InnerError,
            r".*reduce\(\) of empty iterable with no initial value.*",
            try_reduce_iter_failed,
            f,
            l,
        )

    def test_reduce_with_init_value(self):
        self.assert_results(try_reduce, lambda acc, x: acc + x, [1, 3, 4], -2)

    def test_reduce_with_builtin_fn(self):
        self.assert_results(try_reduce, operator.add, [2, 5, 8])


if __name__ == "__main__":
    unittest.main()
