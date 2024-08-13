# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
from dataclasses import dataclass

from test_case_base import TestCaseBase

import paddle
from paddle.jit.sot.utils import strict_mode_guard


@dataclass
class Data:
    x: paddle.Tensor


@dataclass
class DataWithPostInit:
    x: paddle.Tensor

    def __post_init__(self):
        self.x += 1


def return_dataclass(x):
    return Data(x + 1)


def return_dataclass_with_post_init(x):
    return DataWithPostInit(x)


class TestDataclass(TestCaseBase):
    @strict_mode_guard(False)
    def test_dtype_reconstruct(self):
        x = paddle.to_tensor(1)
        self.assert_results(return_dataclass, x)

    @strict_mode_guard(False)
    def test_dtype_reconstruct_with_post_init(self):
        x = paddle.to_tensor(1)
        self.assert_results(return_dataclass_with_post_init, x)


if __name__ == "__main__":
    unittest.main()
