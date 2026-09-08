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

# [AUTO-GENERATED]
# Target file: paddle/phi/kernels/cpu/atan2_kernel.cc
# Tests for atan2 CPU kernel.
# Exercises the C++ Atan2Kernel via paddle.atan2 API.
#
# 本文件针对 atan2_kernel.cc 中的 atan2 CPU 算子编写单元测试。
# 通过 paddle.atan2 API 来调用 C++ 内核，验证四象限反正切计算结果。

import unittest

import numpy as np

import paddle


class TestAtan2CPU(unittest.TestCase):
    """Test atan2 on CPU.
    测试 CPU 上的 atan2 四象限反正切函数。"""

    def setUp(self):
        paddle.set_device("cpu")

    def tearDown(self):
        pass

    def test_atan2_basic(self):
        """Basic atan2 test: atan2(1, 1) = pi/4.
        基础 atan2 测试：atan2(1, 1) = pi/4。"""
        y = paddle.to_tensor([1.0])
        x = paddle.to_tensor([1.0])
        result = paddle.atan2(y, x)
        np.testing.assert_array_almost_equal(result.numpy(), [np.pi / 4])

    def test_atan2_first_quadrant(self):
        """Atan2 in first quadrant.
        第一象限的 atan2 测试。"""
        y = paddle.to_tensor([1.0, 1.0, 3.0])
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        result = paddle.atan2(y, x)
        expected = np.arctan2([1.0, 1.0, 3.0], [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_atan2_second_quadrant(self):
        """Atan2 in second quadrant (y>0, x<0).
        第二象限的 atan2 测试（y>0, x<0）。"""
        y = paddle.to_tensor([1.0, 1.0])
        x = paddle.to_tensor([-1.0, -2.0])
        result = paddle.atan2(y, x)
        expected = np.arctan2([1.0, 1.0], [-1.0, -2.0])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_atan2_third_quadrant(self):
        """Atan2 in third quadrant (y<0, x<0).
        第三象限的 atan2 测试（y<0, x<0）。"""
        y = paddle.to_tensor([-1.0, -2.0])
        x = paddle.to_tensor([-1.0, -1.0])
        result = paddle.atan2(y, x)
        expected = np.arctan2([-1.0, -2.0], [-1.0, -1.0])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_atan2_fourth_quadrant(self):
        """Atan2 in fourth quadrant (y<0, x>0).
        第四象限的 atan2 测试（y<0, x>0）。"""
        y = paddle.to_tensor([-1.0, -3.0])
        x = paddle.to_tensor([1.0, 1.0])
        result = paddle.atan2(y, x)
        expected = np.arctan2([-1.0, -3.0], [1.0, 1.0])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_atan2_broadcast(self):
        """Atan2 with broadcasting.
        带有广播机制的 atan2 测试。"""
        y = paddle.to_tensor([[1.0], [2.0]])  # shape [2, 1]
        x = paddle.to_tensor([1.0, -1.0, 0.0])  # shape [3]
        result = paddle.atan2(y, x)
        self.assertEqual(result.shape, [2, 3])
        for i in range(2):
            for j in range(3):
                np.testing.assert_almost_equal(
                    result[i, j].item(),
                    np.arctan2(y[i, 0].item(), x[j].item()),
                )

    def test_atan2_float64(self):
        """Atan2 with float64 dtype.
        float64 数据类型的 atan2 测试。"""
        y = paddle.to_tensor([1.0, 2.0], dtype="float64")
        x = paddle.to_tensor([1.0, 2.0], dtype="float64")
        result = paddle.atan2(y, x)
        self.assertEqual(result.dtype, paddle.float64)
        expected = np.arctan2([1.0, 2.0], [1.0, 2.0])
        np.testing.assert_array_almost_equal(result.numpy(), expected)

    def test_atan2_positive_y_zero_x(self):
        """Atan2(positive, 0) = pi/2.
        atan2(正数, 0) = pi/2。"""
        y = paddle.to_tensor([1.0, 2.0])
        x = paddle.to_tensor([0.0, 0.0])
        result = paddle.atan2(y, x)
        np.testing.assert_array_almost_equal(
            result.numpy(), [np.pi / 2, np.pi / 2]
        )

    def test_atan2_zero_inputs(self):
        """Atan2(0, 0) = 0.
        atan2(0, 0) = 0。"""
        y = paddle.to_tensor([0.0])
        x = paddle.to_tensor([0.0])
        result = paddle.atan2(y, x)
        np.testing.assert_array_almost_equal(result.numpy(), [0.0])

    def test_atan2_2d(self):
        """Atan2 with 2D tensors.
        二维张量的 atan2 测试。"""
        y = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0]])
        x = paddle.to_tensor([[1.0, -1.0], [-1.0, 1.0]])
        result = paddle.atan2(y, x)
        expected = np.arctan2(
            [[1.0, 2.0], [3.0, 4.0]], [[1.0, -1.0], [-1.0, 1.0]]
        )
        np.testing.assert_array_almost_equal(result.numpy(), expected)


if __name__ == "__main__":
    unittest.main()
