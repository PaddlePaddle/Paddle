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

import unittest

from test_case_base import TestCaseBase

import paddle


def func_dup_top_1():
    return True == True != False


def func_dup_top_2(x):
    y = x + 1
    return True == True != False


def func_dup_top_two(x: list[paddle.Tensor]):
    x[0] += x[1]
    return x


class TestDupTop(TestCaseBase):
    def test_dup_top(self):
        self.assert_results(func_dup_top_1)
        self.assert_results(func_dup_top_2, paddle.to_tensor(1.0))
        # TODO: fix this after we support side effect
        # self.assert_results(
        #     func_dup_top_two, [paddle.to_tensor(1.0), paddle.to_tensor(2.0)]
        # )


if __name__ == "__main__":
    unittest.main()
