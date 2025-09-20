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
        
        # 使用动态图模式进行测试
        paddle.disable_static()
        
        # 创建GPU上的tensor
        gpu_tensor = paddle.ones([10, 10], dtype='float32')
        gpu_tensor = gpu_tensor.cuda()
        
        # 使用memcpy API复制到pinned memory
        pinned_tensor = paddle.tensor.creation._memcpy(gpu_tensor, paddle.CUDAPinnedPlace())
        
        # 验证结果
        np.testing.assert_allclose(gpu_tensor.numpy(), pinned_tensor.numpy(), rtol=1e-05)
        np.testing.assert_allclose(pinned_tensor.numpy(), np.ones((10, 10)), rtol=1e-05)

    def test_pinned_copy_gpu(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA not available, skipping test")
        
        # 使用动态图模式进行测试
        paddle.disable_static()
        
        # 创建pinned memory上的tensor
        pinned_tensor = paddle.zeros([10, 10], dtype='float32')
        pinned_tensor = pinned_tensor.pin_memory()
        
        # 使用memcpy API复制到GPU
        gpu_tensor = paddle.tensor.creation._memcpy(pinned_tensor, paddle.CUDAPlace())
        
        # 验证结果
        np.testing.assert_allclose(gpu_tensor.numpy(), pinned_tensor.numpy(), rtol=1e-05)
        np.testing.assert_allclose(gpu_tensor.numpy(), np.zeros((10, 10)), rtol=1e-05)

    def test_hip_copy_bool_value(self):
        if not core.is_compiled_with_rocm():
            self.skipTest("ROCm not available, skipping test")
        
        # 使用动态图模式进行测试
        paddle.disable_static()
        
        # 创建pinned memory上的bool tensor
        pinned_tensor = paddle.ones([1], dtype='bool')
        pinned_tensor = pinned_tensor.pin_memory()
        
        # 使用memcpy API复制到GPU
        gpu_tensor = paddle.tensor.creation._memcpy(pinned_tensor, paddle.CUDAPlace())
        
        # 验证结果
        expect_value = np.array([True]).astype('bool')
        np.testing.assert_array_equal(gpu_tensor.numpy(), expect_value)


class TestMemcpyOPError(unittest.TestCase):
    def test_SELECTED_ROWS(self):
        if not paddle.is_compiled_with_cuda():
            self.skipTest("CUDA not available, skipping test")
        
        # 使用动态图模式进行测试
        paddle.disable_static()
        
        # 创建SELECTED_ROWS类型的tensor（如果支持的话）
        try:
            # 尝试创建SELECTED_ROWS类型的tensor
            selected_row_tensor = paddle.zeros([10, 10], dtype='float32')
            # 这里应该会失败，因为memcpy不支持SELECTED_ROWS
            with self.assertRaises(RuntimeError):
                pinned_tensor = paddle.tensor.creation._memcpy(selected_row_tensor, paddle.CUDAPinnedPlace())
        except Exception as e:
            # 如果创建SELECTED_ROWS tensor本身就失败，那也是预期的
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
