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

import os
import tempfile
import unittest

import numpy as np

import paddle
from paddle.base import core
from paddle.utils import gpu_utils


class TestSaveDenseTensorToNpy(unittest.TestCase):
    def _save_and_load(self, tensor):
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            file_path = tmp.name
        saved_path = file_path
        try:
            core.eager._save_dense_tensor_to_npy(tensor, file_path)
            saved_path = gpu_utils._resolve_npy_path_for_load(file_path)
            return np.load(saved_path)
        finally:
            for path in {file_path, saved_path}:
                if os.path.exists(path):
                    os.remove(path)

    def _save_and_load_tensor(self, tensor):
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            file_path = tmp.name
        saved_path = file_path
        try:
            core.eager._save_dense_tensor_to_npy(tensor, file_path)
            saved_path = gpu_utils._resolve_npy_path_for_load(file_path)
            return saved_path, gpu_utils._load_dense_tensor_from_npy(file_path)
        finally:
            for path in {file_path, saved_path}:
                if os.path.exists(path):
                    os.remove(path)

    def test_float32_2d(self):
        expected = np.array([[1.0, 2.5, -3.0], [4.0, 5.25, 6.0]], 'float32')
        actual = self._save_and_load(paddle.to_tensor(expected))
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, expected.dtype)

    def test_int64_1d(self):
        expected = np.array([1, 2, 3, 4, 5], 'int64')
        actual = self._save_and_load(paddle.to_tensor(expected))
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, expected.dtype)

    def test_bool_1d(self):
        expected = np.array([True, False, True, False], 'bool')
        actual = self._save_and_load(paddle.to_tensor(expected))
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, expected.dtype)

    def test_float16_scalar(self):
        expected = np.array(3.5, 'float16')
        actual = self._save_and_load(paddle.to_tensor(expected))
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.dtype, expected.dtype)

    def test_empty_tensor(self):
        expected = np.empty([0, 3], 'float32')
        actual = self._save_and_load(paddle.to_tensor(expected))
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.shape, expected.shape)

    def test_bfloat16_saves_as_marked_float32(self):
        expected = np.array([[1.0, 2.5, -3.0], [4.0, 5.25, 6.0]], 'float32')
        tensor = paddle.to_tensor(expected, dtype='bfloat16')
        saved_path, actual_tensor = self._save_and_load_tensor(tensor)
        self.assertTrue(saved_path.endswith('.bf16.npy'))
        self.assertEqual(str(actual_tensor.dtype), 'paddle.bfloat16')
        np.testing.assert_allclose(
            actual_tensor.astype('float32').numpy(), expected, rtol=0, atol=0
        )

    def test_load_float32_tensor_from_npy(self):
        expected = np.array([[1.0, 2.5], [3.0, 4.5]], 'float32')
        _, actual_tensor = self._save_and_load_tensor(
            paddle.to_tensor(expected)
        )
        self.assertEqual(str(actual_tensor.dtype), 'paddle.float32')
        np.testing.assert_array_equal(actual_tensor.numpy(), expected)

    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "Requires CUDA compiled Paddle"
    )
    def test_gpu_tensor(self):
        expected = np.arange(6, dtype='float32').reshape([2, 3])
        tensor = paddle.to_tensor(expected).cuda()
        actual = self._save_and_load(tensor)
        np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
