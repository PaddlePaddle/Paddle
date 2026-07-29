# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import paddle

paddle.enable_compat()

import scalar_type_compat_test
import torch


class TestScalarTypeCaster(unittest.TestCase):
    def test_paddle_dtype_round_trip(self):
        expected_values = {
            torch.uint8: 0,
            torch.int8: 1,
            torch.int16: 2,
            torch.int32: 3,
            torch.int64: 4,
            torch.float16: 5,
            torch.float32: 6,
            torch.float64: 7,
            torch.complex64: 9,
            torch.complex128: 10,
            torch.bool: 11,
            torch.bfloat16: 15,
            torch.uint16: 27,
            torch.uint32: 28,
            torch.float8_e4m3fn: 24,
            torch.float8_e5m2: 23,
        }
        for dtype, expected_value in expected_values.items():
            self.assertIs(
                scalar_type_compat_test.scalar_type_round_trip(dtype), dtype
            )
            self.assertEqual(
                scalar_type_compat_test.scalar_type_value(dtype),
                expected_value,
            )


if __name__ == '__main__':
    unittest.main()
