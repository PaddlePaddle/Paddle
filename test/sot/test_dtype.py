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

from test_case_base import (
    TestCaseBase,
    test_instruction_translator_cache_context,
    test_with_faster_guard,
)

import paddle


def tensor_astype(x, y):
    z = x.astype(y.dtype)
    return z


def tensor_dtype_guard(x):
    return x + 1


def reconstruct_dtype():
    x = paddle.to_tensor(1)
    z = x.dtype
    if x > 0:
        y = paddle.to_tensor(1, dtype=z)
    else:
        y = 1
    return y


def dtype_guard(x, cast_map):
    out = paddle.cast(x, cast_map[x.dtype])
    return out, out.dtype


class TestTensorAstype(TestCaseBase):
    def test_tensor_astype(self):
        x = paddle.ones([2, 3], dtype="float32")
        y = paddle.ones([2, 3], dtype="int32")
        self.assert_results(tensor_astype, x, y)


class TestTensorDtypeGuard(TestCaseBase):
    @test_with_faster_guard
    def test_tensor_dtype_guard(self):
        x = paddle.ones([2, 3], dtype="float32")
        y = paddle.ones([2, 3], dtype="int32")
        with test_instruction_translator_cache_context() as ctx:
            self.assert_results(tensor_dtype_guard, x)
            self.assertEqual(ctx.translate_count, 1)
            self.assert_results(tensor_dtype_guard, y)
            self.assertEqual(ctx.translate_count, 2)
            self.assert_results(tensor_dtype_guard, x)
            self.assertEqual(ctx.translate_count, 2)


class TestDtypeReconstruct(TestCaseBase):
    def test_dtype_reconstruct(self):
        self.assert_results(reconstruct_dtype)


class TestDtypeGuard(TestCaseBase):
    @test_with_faster_guard
    def test_dtype_guard(self):
        dtype_map = {paddle.float32: paddle.float64}
        x = paddle.ones([2, 3], dtype="float32")
        self.assert_results(dtype_guard, x, dtype_map)


if __name__ == "__main__":
    unittest.main()
