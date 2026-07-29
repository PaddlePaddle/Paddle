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

# ruff: noqa: I001

import unittest
from pathlib import Path

import paddle

paddle.enable_compat()

import torch
from paddle.utils.cpp_extension import load


paddle_root = Path(__file__).resolve().parents[2]
extra_include_paths = [
    str(paddle_root),
    str(paddle_root / 'paddle/phi/api/include/compat'),
    str(paddle_root / 'paddle/phi/api/include/compat/torch/csrc/api/include'),
]
build_root = Path(paddle.base.libpaddle.__file__).resolve().parents[3]
build_pybind_include = (
    build_root / 'third_party/pybind/src/extern_pybind/include'
)
if build_pybind_include.is_dir():
    extra_include_paths.insert(1, str(build_pybind_include))

scalar_type_extension = load(
    name='scalar_type_compat_test',
    sources=[str(Path(__file__).with_name('scalar_type_extension.cc'))],
    extra_include_paths=extra_include_paths,
    verbose=True,
)


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
        }
        for dtype, expected_value in expected_values.items():
            self.assertIs(
                scalar_type_extension.scalar_type_round_trip(dtype), dtype
            )
            self.assertEqual(
                scalar_type_extension.scalar_type_value(dtype),
                expected_value,
            )


if __name__ == '__main__':
    unittest.main()
