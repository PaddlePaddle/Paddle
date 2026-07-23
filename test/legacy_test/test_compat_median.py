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

import numpy as np

import paddle
from paddle import base


class TestCompatMedianAPI(unittest.TestCase):
    def test_compat_median_basic(self):
        paddle.disable_static()

        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='float32')

        result = paddle.compat.median(x)
        expected = paddle.to_tensor(5, dtype='float32')
        np.testing.assert_allclose(result.numpy(), expected.numpy())

        values, indices = paddle.compat.median(x, dim=1)
        expected_values = paddle.to_tensor([2, 5, 8], dtype='float32')
        expected_indices = paddle.to_tensor([1, 1, 1], dtype='int64')
        np.testing.assert_allclose(values.numpy(), expected_values.numpy())
        np.testing.assert_allclose(indices.numpy(), expected_indices.numpy())

        result = paddle.compat.median(x, dim=1)
        np.testing.assert_allclose(
            result.values.numpy(), expected_values.numpy()
        )
        np.testing.assert_allclose(
            result.indices.numpy(), expected_indices.numpy()
        )

        values, indices = paddle.compat.median(x, dim=1, keepdim=True)
        expected_values = paddle.to_tensor([[2], [5], [8]], dtype='float32')
        expected_indices = paddle.to_tensor([[1], [1], [1]], dtype='int64')
        np.testing.assert_allclose(values.numpy(), expected_values.numpy())
        np.testing.assert_allclose(indices.numpy(), expected_indices.numpy())

        paddle.enable_static()

    def test_compat_median_out(self):
        paddle.disable_static()

        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='float32')

        out = paddle.zeros([], dtype='float32')
        result = paddle.compat.median(x, out=out)
        expected = paddle.to_tensor(5, dtype='float32')
        np.testing.assert_allclose(result.numpy(), expected.numpy())
        np.testing.assert_allclose(out.numpy(), expected.numpy())
        self.assertIs(result, out)

        out_values = paddle.zeros([3], dtype='float32')
        out_indices = paddle.zeros([3], dtype='int64')
        result_values, result_indices = paddle.compat.median(
            x, dim=1, out=(out_values, out_indices)
        )
        expected_values = paddle.to_tensor([2, 5, 8], dtype='float32')
        expected_indices = paddle.to_tensor([1, 1, 1], dtype='int64')
        np.testing.assert_allclose(
            result_values.numpy(), expected_values.numpy()
        )
        np.testing.assert_allclose(
            result_indices.numpy(), expected_indices.numpy()
        )
        np.testing.assert_allclose(out_values.numpy(), expected_values.numpy())
        np.testing.assert_allclose(
            out_indices.numpy(), expected_indices.numpy()
        )
        self.assertIs(result_values, out_values)
        self.assertIs(result_indices, out_indices)

        paddle.enable_static()

    def test_compat_median_different_dims(self):
        paddle.disable_static()

        x = paddle.to_tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='float32')

        values, indices = paddle.compat.median(x, dim=0)
        expected_values = paddle.to_tensor([4, 5, 6], dtype='float32')
        expected_indices = paddle.to_tensor([1, 1, 1], dtype='int64')
        np.testing.assert_allclose(values.numpy(), expected_values.numpy())
        np.testing.assert_allclose(indices.numpy(), expected_indices.numpy())

        values, indices = paddle.compat.median(x, dim=1)
        expected_values = paddle.to_tensor([2, 5, 8], dtype='float32')
        expected_indices = paddle.to_tensor([1, 1, 1], dtype='int64')
        np.testing.assert_allclose(values.numpy(), expected_values.numpy())
        np.testing.assert_allclose(indices.numpy(), expected_indices.numpy())

        values, indices = paddle.compat.median(x, dim=-1)
        expected_values = paddle.to_tensor([2, 5, 8], dtype='float32')
        expected_indices = paddle.to_tensor([1, 1, 1], dtype='int64')
        np.testing.assert_allclose(values.numpy(), expected_values.numpy())
        np.testing.assert_allclose(indices.numpy(), expected_indices.numpy())

        paddle.enable_static()

    def test_tie_gradient_empty_axis_and_cpu_dtypes(self):
        paddle.disable_static()
        x = paddle.to_tensor(
            [[1.0, 2.0, 3.0, 2.0, 1.0, 3.0]],
            stop_gradient=False,
        )
        result = paddle.compat.median(x, dim=1)
        np.testing.assert_array_equal(result.indices.numpy(), [1])
        result.values.backward()
        np.testing.assert_array_equal(
            x.grad.numpy(), [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
        )

        x = paddle.to_tensor([0.0, 0.0, 0.0], stop_gradient=False)
        result = paddle.compat.median(x, dim=0)
        self.assertEqual(result.indices.item(), 1)
        result.values.backward()
        np.testing.assert_array_equal(x.grad.numpy(), [0.0, 1.0, 0.0])

        x = paddle.to_tensor([0.0, -0.0, 1.0], stop_gradient=False)
        global_result = paddle.compat.median(x)
        self.assertFalse(paddle.signbit(global_result).item())
        global_result.backward()
        np.testing.assert_array_equal(x.grad.numpy(), [0.5, 0.5, 0.0])
        x.clear_gradient()
        result = paddle.compat.median(x, dim=0)
        self.assertEqual(result.indices.item(), 1)
        self.assertTrue(paddle.signbit(result.values).item())
        result.values.backward()
        np.testing.assert_array_equal(x.grad.numpy(), [0.0, 1.0, 0.0])

        matrix = paddle.to_tensor([[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])
        for dim in (np.int64(1), paddle.to_tensor(1, dtype='int8')):
            result = paddle.compat.median(matrix, dim=dim)
            np.testing.assert_array_equal(result.values.numpy(), [2.0, 5.0])
            np.testing.assert_array_equal(result.indices.numpy(), [2, 2])

        for dtype in ('float16', 'int8', 'uint8', 'int16'):
            result = paddle.compat.median(
                paddle.to_tensor([[3, 1, 2]], dtype=dtype), dim=1
            )
            self.assertEqual(result.values.dtype, getattr(paddle, dtype))
            np.testing.assert_array_equal(result.values.numpy(), [2])

        with self.assertRaises(IndexError):
            paddle.compat.median(paddle.empty([2, 0]), dim=1)

        scalar = paddle.to_tensor(3.0, stop_gradient=False)
        for dim in (-1, 0):
            result = paddle.compat.median(scalar, dim=dim)
            self.assertEqual(result.values.shape, [])
            self.assertEqual(result.indices.item(), 0)
            result.values.backward()
            self.assertEqual(scalar.grad.item(), 1.0)
            scalar.clear_gradient()

    def test_compat_median_static(self):
        paddle.enable_static()

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.static.data(name='x', shape=[3, 3], dtype='float32')
            values, indices = paddle.compat.median(x, dim=1)

            exe = base.Executor(base.CPUPlace())
            x_data = np.array(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='float32'
            )
            result_values, result_indices = exe.run(
                feed={'x': x_data}, fetch_list=[values, indices]
            )

            expected_values = np.array([2, 5, 8], dtype='float32')
            expected_indices = np.array([1, 1, 1], dtype='int64')
            np.testing.assert_allclose(result_values, expected_values)
            np.testing.assert_allclose(result_indices, expected_indices)

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.static.data(name='x', shape=[3, 3], dtype='float32')
            result = paddle.compat.median(x, dim=1)

            exe = base.Executor(base.CPUPlace())
            x_data = np.array(
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype='float32'
            )
            result_values, result_indices = exe.run(
                feed={'x': x_data}, fetch_list=[result.values, result.indices]
            )

            expected_values = np.array([2, 5, 8], dtype='float32')
            expected_indices = np.array([1, 1, 1], dtype='int64')
            np.testing.assert_allclose(result_values, expected_values)
            np.testing.assert_allclose(result_indices, expected_indices)

        paddle.disable_static()


class TestCompatNanmedianAPI(unittest.TestCase):
    def test_compat_nanmedian_basic(self):
        paddle.disable_static()

        x = paddle.to_tensor(
            [[1, float('nan'), 3], [4, 5, 6], [float('nan'), 8, 9]],
            dtype='float32',
        )

        result = paddle.compat.nanmedian(x)
        expected = paddle.to_tensor(5.0, dtype='float32')
        np.testing.assert_allclose(result.numpy(), expected.numpy())

        values, indices = paddle.compat.nanmedian(x, dim=1)
        expected_values = paddle.to_tensor([1.0, 5.0, 8.0], dtype='float32')
        expected_indices = paddle.to_tensor([0, 1, 1], dtype='int64')
        np.testing.assert_allclose(values.numpy(), expected_values.numpy())
        np.testing.assert_allclose(indices.numpy(), expected_indices.numpy())

        result = paddle.compat.nanmedian(x, dim=1)
        np.testing.assert_allclose(
            result.values.numpy(), expected_values.numpy()
        )
        np.testing.assert_allclose(
            result.indices.numpy(), expected_indices.numpy()
        )

        values, indices = paddle.compat.nanmedian(x, dim=-1)
        expected_values = paddle.to_tensor([1.0, 5.0, 8.0], dtype='float32')
        expected_indices = paddle.to_tensor([0, 1, 1], dtype='int64')
        np.testing.assert_allclose(values.numpy(), expected_values.numpy())
        np.testing.assert_allclose(indices.numpy(), expected_indices.numpy())

        values, indices = paddle.compat.nanmedian(x, dim=1, keepdim=True)
        expected_values = paddle.to_tensor(
            [[1.0], [5.0], [8.0]], dtype='float32'
        )
        expected_indices = paddle.to_tensor([[0], [1], [1]], dtype='int64')
        np.testing.assert_allclose(values.numpy(), expected_values.numpy())
        np.testing.assert_allclose(indices.numpy(), expected_indices.numpy())

        paddle.enable_static()

    def test_compat_nanmedian_out(self):
        paddle.disable_static()

        x = paddle.to_tensor(
            [[1, float('nan'), 3], [4, 5, 6], [float('nan'), 8, 9]],
            dtype='float32',
        )

        out = paddle.zeros([], dtype='float32')
        result = paddle.compat.nanmedian(x, out=out)
        expected = paddle.to_tensor(5.0, dtype='float32')
        np.testing.assert_allclose(result.numpy(), expected.numpy())
        np.testing.assert_allclose(out.numpy(), expected.numpy())
        self.assertIs(result, out)

        out_values = paddle.zeros([3], dtype='float32')
        out_indices = paddle.zeros([3], dtype='int64')
        result_values, result_indices = paddle.compat.nanmedian(
            x, dim=1, out=(out_values, out_indices)
        )
        expected_values = paddle.to_tensor([1.0, 5.0, 8.0], dtype='float32')
        expected_indices = paddle.to_tensor([0, 1, 1], dtype='int64')
        np.testing.assert_allclose(
            result_values.numpy(), expected_values.numpy()
        )
        np.testing.assert_allclose(
            result_indices.numpy(), expected_indices.numpy()
        )
        np.testing.assert_allclose(out_values.numpy(), expected_values.numpy())
        np.testing.assert_allclose(
            out_indices.numpy(), expected_indices.numpy()
        )
        self.assertIs(result_values, out_values)
        self.assertIs(result_indices, out_indices)

        paddle.enable_static()

    def test_compat_nanmedian_all_nan(self):
        paddle.disable_static()

        x = paddle.to_tensor(
            [[1, 2, 3], [float('nan'), float('nan'), float('nan')], [7, 8, 9]],
            dtype='float32',
            stop_gradient=False,
        )

        values, indices = paddle.compat.nanmedian(x, dim=1)
        expected_values = paddle.to_tensor(
            [2.0, float('nan'), 8.0], dtype='float32'
        )
        expected_indices = paddle.to_tensor([1, 0, 1], dtype='int64')
        np.testing.assert_allclose(
            values.numpy(), expected_values.numpy(), equal_nan=True
        )
        np.testing.assert_allclose(indices.numpy(), expected_indices.numpy())
        values.sum().backward()
        np.testing.assert_array_equal(
            x.grad.numpy(),
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )

        paddle.enable_static()

    def test_tie_gradient_empty_axis_and_cpu_dtype(self):
        paddle.disable_static()
        x = paddle.to_tensor(
            [[1.0, 2.0, float('nan'), 2.0, 3.0]],
            stop_gradient=False,
        )
        result = paddle.compat.nanmedian(x, dim=1)
        np.testing.assert_array_equal(result.indices.numpy(), [1])
        result.values.backward()
        np.testing.assert_array_equal(
            x.grad.numpy(), [[0.0, 1.0, 0.0, 0.0, 0.0]]
        )

        x = paddle.to_tensor(
            [[4.0, 5.0, 6.0, 5.0, float('nan'), 6.0]],
            stop_gradient=False,
        )
        result = paddle.compat.nanmedian(x, dim=1)
        np.testing.assert_array_equal(result.indices.numpy(), [3])
        result.values.backward()
        np.testing.assert_array_equal(
            x.grad.numpy(), [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
        )

        result = paddle.compat.nanmedian(
            paddle.to_tensor([[3, 1, 2]], dtype='int8'), dim=1
        )
        self.assertEqual(result.values.dtype, paddle.int8)
        with self.assertRaises(IndexError):
            paddle.compat.nanmedian(paddle.empty([2, 0]), dim=1)

        scalar = paddle.to_tensor(3.0, stop_gradient=False)
        for dim in (-1, 0):
            result = paddle.compat.nanmedian(scalar, dim=dim)
            self.assertEqual(result.values.shape, [])
            self.assertEqual(result.indices.item(), 0)
            result.values.backward()
            self.assertEqual(scalar.grad.item(), 1.0)
            scalar.clear_gradient()

    def test_compat_nanmedian_static(self):
        paddle.enable_static()

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.static.data(name='x', shape=[3, 3], dtype='float32')
            values, indices = paddle.compat.nanmedian(x, dim=1)

            exe = base.Executor(base.CPUPlace())
            x_data = np.array(
                [[1, float('nan'), 3], [4, 5, 6], [float('nan'), 8, 9]],
                dtype='float32',
            )
            result_values, result_indices = exe.run(
                feed={'x': x_data}, fetch_list=[values, indices]
            )

            expected_values = np.array([1.0, 5.0, 8.0], dtype='float32')
            expected_indices = np.array([0, 1, 1], dtype='int64')
            np.testing.assert_allclose(result_values, expected_values)
            np.testing.assert_allclose(result_indices, expected_indices)

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.static.data(name='x', shape=[3, 3], dtype='float32')
            result = paddle.compat.nanmedian(x, dim=1)

            exe = base.Executor(base.CPUPlace())
            x_data = np.array(
                [[1, float('nan'), 3], [4, 5, 6], [float('nan'), 8, 9]],
                dtype='float32',
            )
            result_values, result_indices = exe.run(
                feed={'x': x_data}, fetch_list=[result.values, result.indices]
            )

            expected_values = np.array([1.0, 5.0, 8.0], dtype='float32')
            expected_indices = np.array([0, 1, 1], dtype='int64')
            np.testing.assert_allclose(result_values, expected_values)
            np.testing.assert_allclose(result_indices, expected_indices)

        paddle.disable_static()


if __name__ == '__main__':
    unittest.main()
