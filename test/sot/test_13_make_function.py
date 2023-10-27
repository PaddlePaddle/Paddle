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

# MAKE_FUNCTION
# CALL_FUNCTION_KW
from __future__ import annotations

import unittest

from test_case_base import TestCaseBase

import paddle


def make_fn(x: paddle.Tensor):
    def fn(a, b=2, c=3, d=4):
        return a + b + c + d

    return fn(1) + fn(2, c=5) + x


class TestExecutor(TestCaseBase):
    def test_simple(self):
        self.assert_results(make_fn, paddle.to_tensor(1))


if __name__ == "__main__":
    unittest.main()
