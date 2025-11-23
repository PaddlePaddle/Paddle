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

from dygraph_to_static_utils import (
    BackendMode,
    Dy2StTestBase,
    ToStaticMode,
    disable_test_case,
)

import paddle
from paddle.jit.api import to_static


def func_test_to_static():
    x = paddle.to_tensor([1, 2, 3])
    return x.get_device()


class TestGetDevice(Dy2StTestBase):
    @disable_test_case(
        (ToStaticMode.SOT_MGS10, BackendMode.PHI | BackendMode.CINN)
    )
    def test_to_static(self):
        static_func = to_static(func_test_to_static)
        static_result = static_func()
        self.assertEqual(static_result, None)


if __name__ == "__main__":
    unittest.main()
