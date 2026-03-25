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
CPU unit test for L1NormGradKernel
(paddle/phi/kernels/cpu/l1_norm_grad_kernel.cc).

Kernel formula: dX = dout * sign(X)
  sign(x) = +1 if x > 0, -1 if x < 0, 0 if x == 0
"""

import unittest

import numpy as np

import paddle


class TestL1NormGradKernel(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()
        paddle.set_device('cpu')

    def test_basic(self):
        """dX = dout * sign(X); loss = l1_norm(x), dout = 1."""
        x_np = np.array([-3.0, 1.0, -2.0, 4.0], dtype='float32')

        x = paddle.to_tensor(x_np, stop_gradient=False)
        loss = paddle.norm(x, p=1)  # sum(abs(x))
        loss.backward()

        # expected: sign(x) * 1.0
        expected = np.sign(x_np).astype('float32')
        np.testing.assert_allclose(x.grad.numpy(), expected, rtol=1e-6)

    def test_2d_input(self):
        """2-D input flattened inside the kernel; same formula applies."""
        x_np = np.array([[1.0, -2.0], [-3.0, 4.0]], dtype='float32')

        x = paddle.to_tensor(x_np, stop_gradient=False)
        loss = paddle.norm(x, p=1)
        loss.backward()

        expected = np.sign(x_np).astype('float32')
        np.testing.assert_allclose(x.grad.numpy(), expected, rtol=1e-6)

    def test_scaled_dout(self):
        """Wrap l1_norm in a chain to get dout != 1: loss = 3 * l1_norm(x)."""
        x_np = np.array([-1.0, 2.0, -3.0], dtype='float32')

        x = paddle.to_tensor(x_np, stop_gradient=False)
        loss = 3.0 * paddle.norm(x, p=1)
        loss.backward()

        expected = 3.0 * np.sign(x_np).astype('float32')
        np.testing.assert_allclose(x.grad.numpy(), expected, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
