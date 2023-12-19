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


def foo(x: paddle.Tensor):
    return x[:, 0]


class TestExecutor(TestCaseBase):
    def test_tensor_slice(self):
        x = paddle.randn((10, 10))
        self.assert_results(foo, x)


if __name__ == "__main__":
    unittest.main()
