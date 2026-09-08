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

# [AUTO-GENERATED] Tests for phi/kernels/cpu/norm_kernel.cc and phi/kernels/cpu/kthvalue_grad_kernel.cc
# norm_kernel.cc: CPU normalize kernel (x / sqrt(sum(x^2) + eps)) along axis
# kthvalue_grad_kernel.cc: CPU gradient of kthvalue op (passes grad to kth element)

import unittest

import numpy as np

import paddle


class TestNormKernel(unittest.TestCase):
    """Test suite for paddle.nn.functional.normalize (norm_kernel.cc) CPU kernel.

    测试 paddle.nn.functional.normalize 的 CPU 内核，涵盖不同轴、数据类型、epsilon 等场景。
    该内核计算 x / sqrt(sum(x^2) + epsilon) 沿指定轴。
    """

    def setUp(self):
        paddle.set_device('cpu')

    def _compute_expected_normalize(self, x_np, axis, epsilon):
        """Helper to compute expected normalize result using numpy.

        使用 numpy 计算预期的归一化结果。
        """
        norm = np.sqrt(np.sum(x_np * x_np, axis=axis, keepdims=True) + epsilon)
        return x_np / norm

    def test_normalize_axis1_float32(self):
        """Test normalize along axis 1 with float32.

        测试 float32 类型沿 axis=1 的归一化。
        """
        x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        x = paddle.to_tensor(x_np)
        result = paddle.nn.functional.normalize(x, p=2, axis=1, epsilon=1e-5)
        expected = self._compute_expected_normalize(x_np, axis=1, epsilon=1e-5)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_normalize_axis0(self):
        """Test normalize along axis 0.

        测试沿 axis=0 的归一化。
        """
        x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        x = paddle.to_tensor(x_np)
        result = paddle.nn.functional.normalize(x, p=2, axis=0, epsilon=1e-5)
        expected = self._compute_expected_normalize(x_np, axis=0, epsilon=1e-5)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_normalize_negative_axis(self):
        """Test normalize with negative axis.

        测试负轴的归一化。
        """
        x_np = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        x = paddle.to_tensor(x_np)
        result = paddle.nn.functional.normalize(x, p=2, axis=-1, epsilon=1e-5)
        expected = self._compute_expected_normalize(x_np, axis=-1, epsilon=1e-5)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_normalize_float64(self):
        """Test normalize with float64 dtype.

        测试 float64 数据类型的归一化。
        """
        x_np = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        x = paddle.to_tensor(x_np)
        result = paddle.nn.functional.normalize(x, p=2, axis=1, epsilon=1e-12)
        expected = self._compute_expected_normalize(x_np, axis=1, epsilon=1e-12)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-10)

    def test_normalize_1d(self):
        """Test normalize on 1D tensor.

        测试 1D 张量的归一化。
        """
        x_np = np.array([3.0, 4.0], dtype=np.float32)
        x = paddle.to_tensor(x_np)
        result = paddle.nn.functional.normalize(x, p=2, axis=0, epsilon=1e-5)
        expected = self._compute_expected_normalize(x_np, axis=0, epsilon=1e-5)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_normalize_3d(self):
        """Test normalize on 3D tensor.

        测试 3D 张量的归一化。
        """
        x_np = np.random.randn(2, 3, 4).astype(np.float32)
        x = paddle.to_tensor(x_np)
        result = paddle.nn.functional.normalize(x, p=2, axis=1, epsilon=1e-5)
        expected = self._compute_expected_normalize(x_np, axis=1, epsilon=1e-5)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-4)

    def test_normalize_unit_vector(self):
        """Test normalize a vector that is already unit length.

        测试已经是单位长度的向量的归一化。
        """
        x_np = np.array([[0.6, 0.8]], dtype=np.float32)  # sqrt(0.36+0.64)=1.0
        x = paddle.to_tensor(x_np)
        result = paddle.nn.functional.normalize(x, p=2, axis=1, epsilon=1e-5)
        np.testing.assert_allclose(result.numpy(), x_np, rtol=1e-5)

    def test_normalize_large_epsilon(self):
        """Test _C_ops.norm with large epsilon (direct kernel test).

        测试 _C_ops.norm 使用较大 epsilon 的情况（直接测试内核）。
        norm_kernel uses: y = x / sqrt(sum(x^2) + eps)
        """
        x = paddle.to_tensor([[3.0, 4.0]], dtype='float32')
        eps = 100.0
        out, norm = paddle._C_ops.norm(x, 1, eps, False)
        expected_norm = np.sqrt(25.0 + 100.0)
        expected_out = np.array([[3.0, 4.0]]) / expected_norm
        np.testing.assert_allclose(norm.numpy(), [[expected_norm]], rtol=1e-5)
        np.testing.assert_allclose(out.numpy(), expected_out, rtol=1e-5)

    def test_normalize_with_zeros(self):
        """Test normalize with zero values in tensor.

        测试张量中包含零值的归一化。
        """
        x_np = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)
        x = paddle.to_tensor(x_np)
        result = paddle.nn.functional.normalize(x, p=2, axis=1, epsilon=1e-5)
        expected = self._compute_expected_normalize(x_np, axis=1, epsilon=1e-5)
        np.testing.assert_allclose(result.numpy(), expected, rtol=1e-5)

    def test_c_ops_norm_returns_norm(self):
        """Test _C_ops.norm returns both normalized output and norm.

        测试 _C_ops.norm 同时返回归一化结果和范数值。
        """
        x = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype='float32')
        out, norm = paddle._C_ops.norm(x, 1, 1e-5, False)
        expected_norm = np.sqrt(
            np.sum(np.array([[1.0, 2.0, 3.0]]) ** 2, axis=1, keepdims=True)
            + 1e-5
        )
        np.testing.assert_allclose(norm.numpy(), expected_norm, rtol=1e-5)
        np.testing.assert_allclose(
            out.numpy(), np.array([[1.0, 2.0, 3.0]]) / expected_norm, rtol=1e-5
        )

    def test_c_ops_norm_is_test(self):
        """Test _C_ops.norm with is_test=True.

        测试 _C_ops.norm 的 is_test=True 模式。
        """
        x = paddle.to_tensor([[1.0, 2.0, 3.0]], dtype='float32')
        out, norm = paddle._C_ops.norm(x, 1, 1e-5, True)
        # is_test=True means norm is a temporary, not stored externally
        # The output should still be correct
        expected_norm = np.sqrt(
            np.sum(np.array([[1.0, 2.0, 3.0]]) ** 2, axis=1, keepdims=True)
            + 1e-5
        )
        np.testing.assert_allclose(
            out.numpy(), np.array([[1.0, 2.0, 3.0]]) / expected_norm, rtol=1e-5
        )


class TestKthvalueGradKernel(unittest.TestCase):
    """Test suite for kthvalue gradient CPU kernel (kthvalue_grad_kernel.cc).

    测试 kthvalue 梯度的 CPU 内核，涵盖不同轴、keepdim、维度等场景。
    梯度内核将梯度传递到第 k 小的元素位置。
    """

    def setUp(self):
        paddle.set_device('cpu')

    def test_kthvalue_grad_1d(self):
        """Test kthvalue gradient on 1D tensor.

        测试 1D 张量的 kthvalue 梯度。
        """
        x = paddle.to_tensor([3.0, 1.0, 4.0, 1.0, 5.0], stop_gradient=False)
        v, i = paddle.kthvalue(x, k=2)
        # k=2: sorted=[1,1,3,4,5], 2nd smallest=1
        # The returned index is the position in the original array
        loss = v.sum()
        loss.backward()
        # Gradient should go to position of the 2nd smallest value
        self.assertAlmostEqual(float(v.numpy()), 1.0, places=5)
        # grad at index i should be 1.0, others 0
        grad = x.grad.numpy()
        self.assertEqual(grad[int(i.numpy())], 1.0)
        self.assertAlmostEqual(float(paddle.sum(x.grad).numpy()), 1.0)

    def test_kthvalue_grad_2d_axis1(self):
        """Test kthvalue gradient on 2D tensor along axis 1.

        测试 2D 张量沿 axis=1 的 kthvalue 梯度。
        """
        x = paddle.to_tensor(
            [[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]], stop_gradient=False
        )
        v, i = paddle.kthvalue(x, k=2, axis=1)
        # Row 0: sorted=[1,2,3], k=2 -> 2 at index 2
        # Row 1: sorted=[4,5,6], k=2 -> 5 at index 2
        loss = v.sum()
        loss.backward()
        expected_grad = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float32)
        np.testing.assert_array_equal(x.grad.numpy(), expected_grad)

    def test_kthvalue_grad_2d_axis0(self):
        """Test kthvalue gradient on 2D tensor along axis 0.

        测试 2D 张量沿 axis=0 的 kthvalue 梯度。
        """
        x = paddle.to_tensor([[3.0, 1.0], [2.0, 4.0]], stop_gradient=False)
        v, i = paddle.kthvalue(x, k=1, axis=0)
        # Col 0: [3,2], k=1 (min) -> 2 at index 1
        # Col 1: [1,4], k=1 (min) -> 1 at index 0
        loss = v.sum()
        loss.backward()
        expected_grad = np.array([[0, 1], [1, 0]], dtype=np.float32)
        np.testing.assert_array_equal(x.grad.numpy(), expected_grad)

    def test_kthvalue_grad_negative_axis(self):
        """Test kthvalue gradient with negative axis.

        测试负轴的 kthvalue 梯度。
        """
        x = paddle.to_tensor(
            [[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]], stop_gradient=False
        )
        v, i = paddle.kthvalue(x, k=2, axis=-1)
        loss = v.sum()
        loss.backward()
        expected_grad = np.array([[0, 0, 1], [0, 0, 1]], dtype=np.float32)
        np.testing.assert_array_equal(x.grad.numpy(), expected_grad)

    def test_kthvalue_grad_keepdim(self):
        """Test kthvalue gradient with keepdim=True.

        测试 keepdim=True 的 kthvalue 梯度。
        """
        x = paddle.to_tensor([[3.0, 1.0, 2.0]], stop_gradient=False)
        v, i = paddle.kthvalue(x, k=2, axis=1, keepdim=True)
        self.assertEqual(v.shape, (1, 1))
        loss = v.sum()
        loss.backward()
        expected_grad = np.array([[0, 0, 1]], dtype=np.float32)
        np.testing.assert_array_equal(x.grad.numpy(), expected_grad)

    def test_kthvalue_grad_k1_min(self):
        """Test kthvalue gradient with k=1 (minimum).

        测试 k=1（最小值）的 kthvalue 梯度。
        """
        x = paddle.to_tensor([5.0, 2.0, 8.0, 1.0], stop_gradient=False)
        v, i = paddle.kthvalue(x, k=1)
        loss = v.sum()
        loss.backward()
        # k=1: minimum is 1.0 at index 3
        self.assertEqual(int(i.numpy()), 3)
        expected_grad = np.array([0, 0, 0, 1], dtype=np.float32)
        np.testing.assert_array_equal(x.grad.numpy(), expected_grad)

    def test_kthvalue_grad_k_max(self):
        """Test kthvalue gradient with k equal to the axis size (maximum).

        测试 k 等于轴长度（最大值）的 kthvalue 梯度。
        """
        x = paddle.to_tensor([5.0, 2.0, 8.0, 1.0], stop_gradient=False)
        v, i = paddle.kthvalue(x, k=4)
        loss = v.sum()
        loss.backward()
        # k=4: maximum is 8.0 at index 2
        self.assertEqual(int(i.numpy()), 2)
        expected_grad = np.array([0, 0, 1, 0], dtype=np.float32)
        np.testing.assert_array_equal(x.grad.numpy(), expected_grad)

    def test_kthvalue_grad_3d(self):
        """Test kthvalue gradient on 3D tensor.

        测试 3D 张量的 kthvalue 梯度。
        """
        x_np = np.array(
            [[[3, 1, 2], [6, 4, 5]], [[9, 7, 8], [12, 10, 11]]],
            dtype=np.float32,
        )
        x = paddle.to_tensor(x_np, stop_gradient=False)
        v, i = paddle.kthvalue(x, k=2, axis=-1)
        loss = v.sum()
        loss.backward()
        # Each row: k=2 means 2nd smallest
        # [3,1,2] -> sorted [1,2,3], k=2 -> 2 at idx 2
        # [6,4,5] -> sorted [4,5,6], k=2 -> 5 at idx 2
        # etc.
        expected_grad = np.zeros_like(x_np)
        expected_grad[0, 0, 2] = 1
        expected_grad[0, 1, 2] = 1
        expected_grad[1, 0, 2] = 1
        expected_grad[1, 1, 2] = 1
        np.testing.assert_array_equal(x.grad.numpy(), expected_grad)

    def test_kthvalue_grad_duplicate_values(self):
        """Test kthvalue gradient with duplicate values.

        测试包含重复值的 kthvalue 梯度。
        """
        x = paddle.to_tensor([1.0, 1.0, 1.0, 2.0], stop_gradient=False)
        v, i = paddle.kthvalue(x, k=2)
        loss = v.sum()
        loss.backward()
        # k=2: sorted=[1,1,1,2], 2nd smallest=1 at first non-zero-diff position
        # Should place gradient at one of the '1' positions
        grad = x.grad.numpy()
        self.assertEqual(int(paddle.sum(x.grad).numpy()), 1)
        self.assertEqual(grad[int(i.numpy())], 1.0)

    def test_kthvalue_grad_scaled(self):
        """Test kthvalue gradient with a scaling factor.

        测试带缩放因子的 kthvalue 梯度。
        """
        x = paddle.to_tensor([3.0, 1.0, 4.0, 1.0, 5.0], stop_gradient=False)
        v, i = paddle.kthvalue(x, k=2)
        loss = v.sum() * 3.14
        loss.backward()
        # Gradient should be 3.14 at the kth position
        grad = x.grad.numpy()
        self.assertAlmostEqual(grad[int(i.numpy())], 3.14, places=3)
        self.assertAlmostEqual(
            float(paddle.sum(x.grad).numpy()), 3.14, places=3
        )


if __name__ == '__main__':
    unittest.main()
