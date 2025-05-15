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

import numpy as np
from dygraph_to_static_utils import (
    Dy2StTestBase,
    test_pir_only,
)

import paddle

SEED = 2025
np.random.seed(SEED)
paddle.seed(SEED)


class TestItem(Dy2StTestBase):
    @classmethod
    def _create_tensor(cls, shape=None, dtype="float32"):
        if shape is None:
            shape = [3, 4, 5]
        return paddle.rand(shape, dtype=dtype)

    def test_no_args(self):
        t = self._create_tensor([1])

        def dynamic_forward(x):
            return x.item()

        static_forward = paddle.jit.to_static(dynamic_forward)
        dynamic_result = dynamic_forward(t)
        static_result = static_forward(t)
        self.assertEqual(dynamic_result, static_result)

    @test_pir_only
    def test_1_arg(self):
        shape_list = [
            [9],
            [3, 5],
            [2, 3, 4],
            [3, 3, 3, 3, 3, 3],
        ]

        for shape in shape_list:
            t = self._create_tensor(shape)

            def dynamic_forward(x):
                return x.item(6)

            static_forward = paddle.jit.to_static(dynamic_forward)
            dynamic_result = dynamic_forward(t)
            static_result = static_forward(t)
            self.assertEqual(dynamic_result, static_result)

    @test_pir_only
    def test_n_arg(self):
        shape_and_idx_list = [
            [[3, 5], [1, 3]],
            [[2, 3, 4], [0, 2, 1]],
            [[2, 3, 4, 5], [0, 1, 3, 0]],
            [[3, 3, 3, 3, 3, 3], [1, 1, 1, 1, 1, 0]],
        ]

        for shape, idx in shape_and_idx_list:
            t = self._create_tensor(shape)

            def dynamic_forward(x, idx):
                return x.item(*idx)

            static_forward = paddle.jit.to_static(dynamic_forward)
            dynamic_result = dynamic_forward(t, idx)
            static_result = static_forward(t, idx)
            self.assertEqual(dynamic_result, static_result)

    @test_pir_only
    def test_error(self):
        def test_raise_error(t, exception_type, expected_exception_str, *args):
            def dynamic_forward(x):
                return x.item(*args)

            static_forward = paddle.jit.to_static(dynamic_forward)

            with self.assertRaisesRegex(exception_type, expected_exception_str):
                static_forward(t)

            with self.assertRaisesRegex(exception_type, expected_exception_str):
                dynamic_forward(t)

        t = self._create_tensor([8, 8, 8], "float32")
        test_raise_error(
            t,
            ValueError,
            "only one element tensors can be converted to "
            "Python scalars when no input coordinates",
        )
        test_raise_error(
            t, ValueError, "index (.)* is out of bounds for size (.)*", 10000
        )
        test_raise_error(
            t, ValueError, "incorrect number of indices for Tensor", 6, 7
        )
        test_raise_error(
            t,
            TypeError,
            r"argument \(position (.)* must be long, but got",
            6.0,
            7.0,
            1.0,
        )
        test_raise_error(
            t,
            ValueError,
            r"index (.)* is out of bounds for axis (.)* with size (.)*",
            9,
            9,
            9,
        )


if __name__ == '__main__':
    unittest.main()
