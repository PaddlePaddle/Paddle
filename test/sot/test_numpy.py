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

import numpy as np
from test_case_base import TestCaseBase

import paddle
from paddle.jit.sot.utils import strict_mode_guard


def foo(x, y):
    ret = x + y
    return ret


class TestNumpy(TestCaseBase):
    def test_tensor_add_numpy_number(self):
        x = paddle.to_tensor([1.0])
        y = np.int64(2)
        self.assert_results(foo, x, y)
        self.assert_results(foo, y, x)

    @strict_mode_guard(False)
    def test_tensor_add_numpy_array(self):
        x = paddle.to_tensor([1.0])
        y = np.array(2.0)
        self.assert_results(foo, x, y)
        self.assert_results(foo, y, x)


if __name__ == "__main__":
    unittest.main()
