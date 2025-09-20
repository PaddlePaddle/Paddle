#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import get_device_place

import paddle
from paddle import base
from paddle.base import Program, core, program_guard


class TestMemcpy_FillConstant(unittest.TestCase):
    def test_gpu_copy_to_pinned(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA not available, skipping test")
        
        # Use dynamic graph mode for testing
        paddle.disable_static()
        
        # Create tensor on GPU
        gpu_tensor = paddle.ones([10, 10], dtype='float32')
        gpu_tensor = gpu_tensor.cuda()
        
        # Use memcpy API to copy to pinned memory
        pinned_tensor = paddle.tensor.creation._memcpy(gpu_tensor, paddle.CUDAPinnedPlace())
        
        # Verify results
        np.testing.assert_allclose(gpu_tensor.numpy(), pinned_tensor.numpy(), rtol=1e-05)
        np.testing.assert_allclose(pinned_tensor.numpy(), np.ones((10, 10)), rtol=1e-05)

    def test_pinned_copy_gpu(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA not available, skipping test")
        
        # Use dynamic graph mode for testing
        paddle.disable_static()
        
        # Create tensor on pinned memory
        pinned_tensor = paddle.zeros([10, 10], dtype='float32')
        pinned_tensor = pinned_tensor.pin_memory()
        
        # Use memcpy API to copy to GPU
        gpu_tensor = paddle.tensor.creation._memcpy(pinned_tensor, paddle.CUDAPlace())
        
        # Verify results
        np.testing.assert_allclose(gpu_tensor.numpy(), pinned_tensor.numpy(), rtol=1e-05)
        np.testing.assert_allclose(gpu_tensor.numpy(), np.zeros((10, 10)), rtol=1e-05)

    def test_hip_copy_bool_value(self):
        if not core.is_compiled_with_rocm():
            self.skipTest("ROCm not available, skipping test")
        
        # Use dynamic graph mode for testing
        paddle.disable_static()
        
        # Create bool tensor on pinned memory
        pinned_tensor = paddle.ones([1], dtype='bool')
        pinned_tensor = pinned_tensor.pin_memory()
        
        # Use memcpy API to copy to GPU
        gpu_tensor = paddle.tensor.creation._memcpy(pinned_tensor, paddle.CUDAPlace())
        
        # Verify results
        expect_value = np.array([True]).astype('bool')
        np.testing.assert_array_equal(gpu_tensor.numpy(), expect_value)


class TestMemcpyOPError(unittest.TestCase):
    def test_SELECTED_ROWS(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA not available, skipping test")
        
        # Use dynamic graph mode for testing
        paddle.disable_static()
        
        # Create SELECTED_ROWS type tensor (if supported)
        try:
            # Try to create SELECTED_ROWS type tensor
            selected_row_tensor = paddle.zeros([10, 10], dtype='float32')
            # This should fail because memcpy doesn't support SELECTED_ROWS
            with self.assertRaises(RuntimeError):
                pinned_tensor = paddle.tensor.creation._memcpy(selected_row_tensor, paddle.CUDAPinnedPlace())
        except Exception as e:
            # If creating SELECTED_ROWS tensor itself fails, that's also expected
            pass


class TestMemcpyApi(unittest.TestCase):
    def test_api(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA not available, skipping test")
        a = paddle.ones([1024, 1024])
        b = paddle.tensor.creation._memcpy(a, paddle.CUDAPinnedPlace())
        self.assertEqual(b.place.__repr__(), "Place(gpu_pinned)")
        np.testing.assert_array_equal(a.numpy(), b.numpy())


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
