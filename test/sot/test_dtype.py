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
    run_in_both_default_and_pir,
    test_instruction_translator_cache_context,
)

import paddle


def tensor_astype(x, y):
    z = x.astype(y.dtype)
    return z


def tensor_dtype_guard(x):
    return x + 1


class TestTensorAstype(TestCaseBase):
    @run_in_both_default_and_pir
    def test_tensor_astype(self):
        x = paddle.ones([2, 3], dtype="float32")
        y = paddle.ones([2, 3], dtype="int32")
        self.assert_results(tensor_astype, x, y)


class TestTensorDtypeGuard(TestCaseBase):
    @run_in_both_default_and_pir
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


if __name__ == "__main__":
    unittest.main()
