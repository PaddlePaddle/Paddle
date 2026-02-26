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

import numpy as np

import paddle
from paddle.compat import clone


class TestCompatClone(unittest.TestCase):
    def setUp(self):
        self.places = [paddle.CPUPlace()]
        if paddle.is_compiled_with_cuda():
            self.places.append(paddle.CUDAPlace(0))

    def test_clone_basic(self):
        for place in self.places:
            with self.subTest(place=place):
                x = paddle.to_tensor([1, 2, 3])
                y = clone(x)

                np.testing.assert_array_equal(x.numpy(), y.numpy())
                self.assertIsNot(x, y)

    def test_clone_with_gradient(self):
        for place in self.places:
            with self.subTest(place=place):
                x = paddle.to_tensor([2.0, 3.0, 4.0], dtype='float32')
                x.stop_gradient = False
                y = clone(x)
                z = y * 2
                z.backward()

                expected_grad = np.array([2.0, 2.0, 2.0], dtype='float32')
                np.testing.assert_array_almost_equal(
                    x.grad.numpy(), expected_grad
                )

    def test_clone_memory_format_ignored(self):
        for place in self.places:
            with self.subTest(place=place):
                x = paddle.to_tensor([1, 2, 3])
                y = clone(x, memory_format='contiguous')

                np.testing.assert_array_equal(x.numpy(), y.numpy())

    def test_clone_memory_format_none(self):
        for place in self.places:
            with self.subTest(place=place):
                x = paddle.to_tensor([1, 2, 3])
                y = clone(x, memory_format=None)

                np.testing.assert_array_equal(x.numpy(), y.numpy())

    def test_clone_multidimensional(self):
        for place in self.places:
            with self.subTest(place=place):
                x = paddle.to_tensor([[1, 2], [3, 4], [5, 6]])
                y = clone(x)

                np.testing.assert_array_equal(x.numpy(), y.numpy())
                self.assertEqual(x.shape, y.shape)

    def test_clone_float_dtype(self):
        for place in self.places:
            with self.subTest(place=place):
                x = paddle.to_tensor([1.5, 2.5, 3.5], dtype='float32')
                y = clone(x)

                np.testing.assert_array_equal(x.numpy(), y.numpy())
                self.assertEqual(x.dtype, y.dtype)

    def test_clone_int_dtype(self):
        for place in self.places:
            with self.subTest(place=place):
                x = paddle.to_tensor([1, 2, 3], dtype='int64')
                y = clone(x)

                np.testing.assert_array_equal(x.numpy(), y.numpy())
                self.assertEqual(x.dtype, y.dtype)

    def test_clone_bool_dtype(self):
        for place in self.places:
            with self.subTest(place=place):
                x = paddle.to_tensor([True, False, True], dtype='bool')
                y = clone(x)

                np.testing.assert_array_equal(x.numpy(), y.numpy())
                self.assertEqual(x.dtype, y.dtype)

    def test_clone_complex_dtype(self):
        for place in self.places:
            with self.subTest(place=place):
                x = paddle.to_tensor([1 + 2j, 3 + 4j], dtype='complex64')
                y = clone(x)

                np.testing.assert_array_equal(x.numpy(), y.numpy())
                self.assertEqual(x.dtype, y.dtype)

    def test_clone_independence(self):
        for place in self.places:
            with self.subTest(place=place):
                x = paddle.to_tensor([1, 2, 3])
                y = clone(x)
                y[0] = 10

                # Original should not be affected
                np.testing.assert_array_equal(x.numpy(), [1, 2, 3])
                np.testing.assert_array_equal(y.numpy(), [10, 2, 3])

    def test_clone_stop_gradient_true(self):
        for place in self.places:
            with self.subTest(place=place):
                x = paddle.to_tensor([1.0, 2.0, 3.0], dtype='float32')
                x.stop_gradient = True
                y = clone(x)

                self.assertTrue(y.stop_gradient)

    def test_clone_stop_gradient_false(self):
        for place in self.places:
            with self.subTest(place=place):
                x = paddle.to_tensor([1.0, 2.0, 3.0], dtype='float32')
                x.stop_gradient = False
                y = clone(x)

                self.assertFalse(y.stop_gradient)

    def test_clone_empty_tensor(self):
        for place in self.places:
            with self.subTest(place=place):
                x = paddle.to_tensor([])
                y = clone(x)

                np.testing.assert_array_equal(x.numpy(), y.numpy())
                self.assertEqual(x.shape, y.shape)

    def test_clone_scalar_tensor(self):
        for place in self.places:
            with self.subTest(place=place):
                x = paddle.to_tensor(5.0)
                y = clone(x)

                np.testing.assert_array_equal(x.numpy(), y.numpy())
                self.assertEqual(x.shape, y.shape)

    def test_clone_with_nan(self):
        for place in self.places:
            with self.subTest(place=place):
                x = paddle.to_tensor([1.0, float('nan'), 3.0], dtype='float32')
                y = clone(x)

                np.testing.assert_array_equal(
                    x.numpy(), y.numpy(), equal_nan=True
                )

    def test_clone_with_inf(self):
        for place in self.places:
            with self.subTest(place=place):
                x = paddle.to_tensor(
                    [1.0, float('inf'), float('-inf')], dtype='float32'
                )
                y = clone(x)

                np.testing.assert_array_equal(x.numpy(), y.numpy())

    def test_clone_large_tensor(self):
        for place in self.places:
            with self.subTest(place=place):
                x = paddle.randn([1000, 1000])
                y = clone(x)

                np.testing.assert_array_equal(x.numpy(), y.numpy())
                self.assertEqual(x.shape, y.shape)

    def test_clone_chain(self):
        for place in self.places:
            with self.subTest(place=place):
                x = paddle.to_tensor([1, 2, 3])
                y = clone(x)
                z = clone(y)

                np.testing.assert_array_equal(x.numpy(), y.numpy())
                np.testing.assert_array_equal(y.numpy(), z.numpy())
                self.assertIsNot(x, y)
                self.assertIsNot(y, z)

    def test_clone_in_computation_graph(self):
        for place in self.places:
            with self.subTest(place=place):
                x = paddle.to_tensor([2.0, 3.0], dtype='float32')
                x.stop_gradient = False
                y = clone(x)
                z = y + 1
                w = clone(z)
                loss = w.sum()
                loss.backward()

                expected_grad = np.array([1.0, 1.0], dtype='float32')
                np.testing.assert_array_almost_equal(
                    x.grad.numpy(), expected_grad
                )


if __name__ == '__main__':
    unittest.main()
