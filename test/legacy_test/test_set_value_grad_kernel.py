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

"""
CPU unit tests for SetValueGradImpl.

Covers the four branches inside SetValueGradImpl
(paddle/phi/kernels/cpu/set_value_grad_kernel.cc):

  Branch 1 - x_grad:
      out_grad is copied to x_grad, then the slice region is zeroed out.

  Branch 2 - value_grad (same shape, need_reverse=false):
      When value_grad.dims() == out_dims and step > 0, the gradient is
      extracted directly via stridedSlice.

  Branch 3 - value_grad (same shape, need_reverse=true):  ** previously uncovered **
      When step < 0, reverse_vector is set and the stridedSlice result is
      reversed before being written to value_grad.

  Branch 4 - value_grad (broadcast/reduce):
      When value_grad.dims() != out_dims, gradient is accumulated across
      multiple broadcast tiles.

The test chain used throughout:
    a = x * x
    a[index] = value * value
    loss = a.sum()
    loss.backward()

Using v*v (instead of a direct assignment) makes gradients non-trivial and
easy to verify analytically:
    d(loss)/d(x[i]) = 2*x[i]   for i NOT in slice region; 0 inside region
    d(loss)/d(v[i]) = 2*v[i]   (each v[i] appears once as v[i]^2)
"""

import unittest

import numpy as np

import paddle


class TestSetValueGradXGrad(unittest.TestCase):
    """
    Branch 1: x_grad path.

    SetValueGradImpl copies out_grad into x_grad, then zeros the slice region.
    Expected: x_grad[i] = 2*x[i] outside the slice; 0 inside.
    """

    def setUp(self):
        paddle.disable_static()
        paddle.set_device('cpu')

    def test_1d_slice_step1(self):
        x_np = np.arange(1.0, 7.0, dtype='float32')  # [1,2,3,4,5,6]
        value_np = np.array([10.0, 10.0], dtype='float32')

        x = paddle.to_tensor(x_np, stop_gradient=False)
        value = paddle.to_tensor(value_np, stop_gradient=False)

        a = x * x
        a[1:3] = value  # slice indices 1,2
        loss = a.sum()
        loss.backward()

        expected = 2.0 * x_np
        expected[1:3] = 0.0

        np.testing.assert_allclose(
            x.grad.numpy(),
            expected,
            rtol=1e-6,
            err_msg=f'x_grad mismatch: expected {expected}, got {x.grad.numpy()}',
        )

    def test_2d_slice(self):
        x_np = np.arange(1.0, 13.0, dtype='float32').reshape(3, 4)
        value_np = np.ones((2, 4), dtype='float32') * 99.0

        x = paddle.to_tensor(x_np, stop_gradient=False)
        value = paddle.to_tensor(value_np, stop_gradient=False)

        a = x * x
        a[0:2, :] = value
        loss = a.sum()
        loss.backward()

        expected = 2.0 * x_np
        expected[0:2, :] = 0.0

        np.testing.assert_allclose(
            x.grad.numpy(),
            expected,
            rtol=1e-6,
        )


class TestSetValueGradValueGradSameShape(unittest.TestCase):
    """
    Branch 2: value_grad when value_grad.dims() == out_dims and step > 0.

    SetValueGradImpl extracts the gradient via a single stridedSlice and assigns
    it directly to value_grad (no accumulation, no reverse).
    Expected: value_grad[i] = 2*v[i].
    """

    def setUp(self):
        paddle.disable_static()
        paddle.set_device('cpu')

    def test_1d_exact_shape(self):
        x_np = np.arange(1.0, 7.0, dtype='float32')
        value_np = np.array([10.0, 20.0, 30.0], dtype='float32')

        x = paddle.to_tensor(x_np, stop_gradient=False)
        value = paddle.to_tensor(value_np, stop_gradient=False)

        a = x * x
        a[2:5] = value * value  # slice output shape [3] == value shape [3]
        loss = a.sum()
        loss.backward()

        expected = 2.0 * value_np
        np.testing.assert_allclose(
            value.grad.numpy(),
            expected,
            rtol=1e-6,
        )

    def test_2d_exact_shape(self):
        x_np = np.arange(1.0, 13.0, dtype='float32').reshape(4, 3)
        value_np = np.arange(100.0, 106.0, dtype='float32').reshape(2, 3)

        x = paddle.to_tensor(x_np, stop_gradient=False)
        value = paddle.to_tensor(value_np, stop_gradient=False)

        a = x * x
        a[1:3, :] = (
            value * value
        )  # slice output shape [2,3] == value shape [2,3]
        loss = a.sum()
        loss.backward()

        expected = 2.0 * value_np
        np.testing.assert_allclose(
            value.grad.numpy(),
            expected,
            rtol=1e-5,
        )


class TestSetValueGradNegativeStep(unittest.TestCase):
    """
    Branch 3: need_reverse = true (step < 0).

    When any axis has a negative step, reverse_vector is set for that axis and
    SetValueGradImpl reverses the stridedSlice result before writing to
    value_grad.  This branch was NOT covered by existing tests.

    The key invariant: value_grad[i] = 2*v[i] regardless of the direction in
    which v was written into x, because the reverse restores the correspondence
    between value positions and gradient positions.
    """

    def setUp(self):
        paddle.disable_static()
        paddle.set_device('cpu')

    def test_1d_step_neg1(self):
        """a[5:2:-1] = v => writes a[5],a[4],a[3]; step=-1 triggers need_reverse."""
        x_np = np.arange(1.0, 7.0, dtype='float32')  # [1,2,3,4,5,6]
        value_np = np.array([10.0, 20.0, 30.0], dtype='float32')

        x = paddle.to_tensor(x_np, stop_gradient=False)
        value = paddle.to_tensor(value_np, stop_gradient=False)

        a = x * x
        a[5:2:-1] = value * value  # fills indices 5,4,3
        loss = a.sum()
        loss.backward()

        # x_grad: 2*x everywhere, 0 at indices {3,4,5}
        expected_x_grad = 2.0 * x_np
        expected_x_grad[3:6] = 0.0

        # value_grad: 2*v (reverse restores original order)
        expected_value_grad = 2.0 * value_np

        np.testing.assert_allclose(
            x.grad.numpy(),
            expected_x_grad,
            rtol=1e-6,
            err_msg=f'x_grad: expected {expected_x_grad}, got {x.grad.numpy()}',
        )
        np.testing.assert_allclose(
            value.grad.numpy(),
            expected_value_grad,
            rtol=1e-6,
            err_msg=f'value_grad: expected {expected_value_grad}, got {value.grad.numpy()}',
        )

    def test_2d_negative_step_axis0(self):
        """a[3:0:-1, :] = v => fills rows 3,2,1 in reverse; step=-1 on axis 0."""
        x_np = np.arange(1.0, 13.0, dtype='float32').reshape(4, 3)
        value_np = np.arange(10.0, 19.0, dtype='float32').reshape(3, 3)

        x = paddle.to_tensor(x_np, stop_gradient=False)
        value = paddle.to_tensor(value_np, stop_gradient=False)

        a = x * x
        a[3:0:-1, :] = value * value  # fills rows 3,2,1
        loss = a.sum()
        loss.backward()

        # x_grad: 2*x for row 0 only; 0 for rows 1,2,3
        expected_x_grad = 2.0 * x_np
        expected_x_grad[1:4, :] = 0.0

        expected_value_grad = 2.0 * value_np

        np.testing.assert_allclose(
            x.grad.numpy(),
            expected_x_grad,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            value.grad.numpy(),
            expected_value_grad,
            rtol=1e-6,
        )

    def test_1d_step_neg2(self):
        """a[::-2] = v => step=-2, hits indices 7,5,3,1."""
        x_np = np.arange(1.0, 9.0, dtype='float32')  # [1..8], shape [8]
        value_np = np.array([10.0, 20.0, 30.0, 40.0], dtype='float32')

        x = paddle.to_tensor(x_np, stop_gradient=False)
        value = paddle.to_tensor(value_np, stop_gradient=False)

        a = x * x
        a[::-2] = value * value  # fills indices 7,5,3,1
        loss = a.sum()
        loss.backward()

        # x_grad=0 at {1,3,5,7}; 2*x elsewhere
        expected_x_grad = 2.0 * x_np
        expected_x_grad[[1, 3, 5, 7]] = 0.0

        expected_value_grad = 2.0 * value_np

        np.testing.assert_allclose(
            x.grad.numpy(),
            expected_x_grad,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            value.grad.numpy(),
            expected_value_grad,
            rtol=1e-6,
        )


class TestSetValueGradValueGradBroadcast(unittest.TestCase):
    """
    Branch 4: value_grad when value_grad.dims() != out_dims (broadcast/reduce).

    SetValueGradImpl accumulates the gradient from multiple broadcast tiles into
    value_grad via repeated additions.
    Expected: value_grad[i] = 2*v[i] * (number of tiles v[i] is broadcast into).
    """

    def setUp(self):
        paddle.disable_static()
        paddle.set_device('cpu')

    def test_scalar_broadcast_into_2d_slice(self):
        """value shape [1] broadcast into slice shape [2,4] (8 elements)."""
        x_np = np.arange(1.0, 13.0, dtype='float32').reshape(3, 4)
        value_np = np.array([5.0], dtype='float32')

        x = paddle.to_tensor(x_np, stop_gradient=False)
        value = paddle.to_tensor(value_np, stop_gradient=False)

        a = x * x
        a[0:2, :] = value * value  # slice [2,4], value [1] -> 8 tiles
        loss = a.sum()
        loss.backward()

        # d(loss)/d(v[0]) = sum over 8 positions of 2*v[0] = 2*5*8 = 80
        expected_value_grad = np.array([2.0 * 5.0 * 8.0], dtype='float32')

        np.testing.assert_allclose(
            value.grad.numpy(),
            expected_value_grad,
            rtol=1e-5,
            err_msg=f'Expected {expected_value_grad}, got {value.grad.numpy()}',
        )

    def test_row_vector_broadcast(self):
        """value shape [1,4] broadcast into slice shape [2,4] (2 rows)."""
        x_np = np.arange(1.0, 13.0, dtype='float32').reshape(3, 4)
        value_np = np.array([[10.0, 20.0, 30.0, 40.0]], dtype='float32')

        x = paddle.to_tensor(x_np, stop_gradient=False)
        value = paddle.to_tensor(value_np, stop_gradient=False)

        a = x * x
        a[0:2, :] = (
            value * value
        )  # slice [2,4], value [1,4] -> each v[j] used 2×
        loss = a.sum()
        loss.backward()

        # d(loss)/d(v[0,j]) = 2*v[0,j] * 2  (2 rows use the same v[0,j])
        expected_value_grad = 2.0 * value_np * 2.0

        np.testing.assert_allclose(
            value.grad.numpy(),
            expected_value_grad,
            rtol=1e-5,
        )


if __name__ == '__main__':
    unittest.main()
