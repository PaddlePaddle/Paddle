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
import unittest

from test_case_base import TestCaseBase

import paddle
from paddle.jit.sot.psdb import check_no_breakgraph


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


class TestFunctools(TestCaseBase):
    def test_lru_cache(self):
        x = paddle.rand([2, 3, 4])
        self.assert_results(fn_with_lru_cache, x)


if __name__ == "__main__":
    unittest.main()
