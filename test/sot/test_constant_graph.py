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

# New Supported Instructions:
# BUILD_MAP (new)
# BUILD_CONST_KEY_MAP (new)

import unittest

from test_case_base import TestCaseBase

import paddle


def func_1(format_str, tensor):
    str = format_str.format(xx=12)
    a = "{xx} = 12".format
    ttt = f"{10} = 12"
    a(xx=12)
    tensor = tensor + 1
    return str, tensor


def func_2(format_str, tensor):
    str = format_str % 10
    tensor = tensor + 1
    return str, tensor


class TestConstantGraph(TestCaseBase):
    def test_case_1(self):
        x = "{xx} is xx"
        tensor = paddle.to_tensor(1)
        self.assert_results(func_1, x, tensor)

    def test_case_2(self):
        x = "%s is xx"
        tensor = paddle.to_tensor(1)
        self.assert_results(func_2, x, tensor)


if __name__ == "__main__":
    unittest.main()
