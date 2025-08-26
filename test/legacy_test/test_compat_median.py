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

        paddle.enable_static()

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
