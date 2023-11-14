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

import re
import unittest

import paddle
from paddle.jit.sot import symbolic_translate


def case1(x):
    return n  # noqa: F821


def case2(x):
    x = x + 1
    return x @ x


def case3(x):
    y = x.undefined_attr
    return y


def case4_inner(x):
    y = x * 2
    print()
    y = y + 1
    return y.undefined_attr


def case4(x):
    return case4_inner(x)


def case5_inner3(x):
    x += 1
    print(x)
    z = x + 1
    return z


def case5_inner2(x):
    x += 1
    z = case5_inner3(1 / 0)
    return z + 1


def case5_inner1(x):
    return case5_inner2(x)


def case5(x):
    y = case5_inner3(x)
    return case5_inner1(y) + 1


class TestException(unittest.TestCase):
    def catch_error(self, func, inputs, error_lines: int | list[int]):
        if isinstance(error_lines, int):
            error_lines = [error_lines]
        try:
            symbolic_translate(func)(inputs)
        except Exception as e:
            match_results = re.compile(r'File ".*", line (\d+)').findall(str(e))
            match_results = list(map(int, match_results))
            assert (
                match_results == error_lines
            ), f"{match_results} is not equal {error_lines}"

    def test_all_case(self):
        self.catch_error(case1, paddle.rand([2, 1]), 25)
        # TODO: support runtime error, such as x[111], x@x
        # self.catch_error(case2, paddle.rand([2, 1]), 30)
        self.catch_error(case3, paddle.rand([2, 1]), 34)
        self.catch_error(case4, paddle.rand([2, 1]), 42)
        self.catch_error(case5, paddle.rand([3, 1]), [68, 63, 58])


if __name__ == "__main__":
    unittest.main()
