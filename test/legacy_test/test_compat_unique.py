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

import paddle
from paddle import base


class TestCompatUniqueAPI(unittest.TestCase):
    def test_basic(self):
        paddle.disable_static()
        x = paddle.to_tensor([2, 3, 3, 1, 5, 3])
        result = paddle.compat.unique(x)
        expected = paddle.to_tensor([1, 2, 3, 5], dtype='int64')
        np.testing.assert_allclose(result.numpy(), expected.numpy())

        _, inverse_indices, counts = paddle.compat.unique(
            x, return_inverse=True, return_counts=True
        )
        expected_indices = paddle.to_tensor([1, 2, 2, 0, 3, 2], dtype='int64')
        expected_counts = paddle.to_tensor([1, 1, 3, 1], dtype='int64')
        np.testing.assert_allclose(
            inverse_indices.numpy(), expected_indices.numpy()
        )
        np.testing.assert_allclose(counts.numpy(), expected_counts.numpy())

        x = paddle.to_tensor([[2, 1, 3], [3, 0, 1], [2, 1, 3]])
        result = paddle.compat.unique(x)
        expected = paddle.to_tensor([0, 1, 2, 3], dtype='int64')
        np.testing.assert_allclose(result.numpy(), expected.numpy())
        paddle.enable_static()

    def test_static(self):
        paddle.enable_static()

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.static.data(name='input', shape=[6], dtype='int64')
            out, inverse_indices, counts = paddle.compat.unique(
                x, return_inverse=True, return_counts=True
            )

            exe = base.Executor(base.CPUPlace())
            x_data = np.array([2, 3, 3, 1, 5, 3], dtype='int64')
            result = exe.run(
                feed={'input': x_data},
                fetch_list=[out, inverse_indices, counts],
            )

            np.testing.assert_allclose(result[1], [1, 2, 2, 0, 3, 2])
            np.testing.assert_allclose(result[2], [1, 1, 3, 1])

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x = paddle.static.data(name='input', shape=[3, 3], dtype='int64')
            out = paddle.compat.unique(x)

            exe = base.Executor(base.CPUPlace())
            x_data = np.array([[2, 1, 3], [3, 0, 1], [2, 1, 3]], dtype='int64')
            result = exe.run(feed={'input': x_data}, fetch_list=[out])

            expected = np.array([0, 1, 2, 3], dtype='int64')
            np.testing.assert_allclose(result[0], expected)

        paddle.disable_static()


if __name__ == '__main__':
    unittest.main()
