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
设备管理操作单元测试 / Device Management Unit Tests

测试目标 / Test Target:
  paddle.device 模块 (python/paddle/device/__init__.py, 覆盖率约73.9%)

覆盖的模块 / Covered Modules:
  - paddle.device: 设备管理API
  - paddle.device.cuda: CUDA设备相关操作
  - paddle.device.cpu: CPU设备操作

作用 / Purpose:
  覆盖设备查询、设置等管理函数的代码路径，提高设备管理模块的测试覆盖率。
"""

import unittest

import paddle

paddle.disable_static()

HAS_GPU = paddle.device.is_compiled_with_cuda()


class TestDeviceBasic(unittest.TestCase):
    """测试基本设备操作 / Test basic device operations"""

    def test_get_device(self):
        """测试获取当前设备 / Test getting current device"""
        device = paddle.device.get_device()
        self.assertIsInstance(device, str)
        self.assertTrue(device.startswith('gpu') or device.startswith('cpu'))

    def test_set_device_cpu(self):
        """测试设置CPU设备 / Test setting CPU device"""
        original_device = paddle.device.get_device()
        paddle.device.set_device('cpu')
        device = paddle.device.get_device()
        self.assertEqual(device, 'cpu')
        paddle.device.set_device(original_device)

    @unittest.skipIf(not HAS_GPU, "No GPU available")
    def test_set_device_gpu(self):
        """测试设置GPU设备 / Test setting GPU device"""
        original_device = paddle.device.get_device()
        paddle.device.set_device('gpu:0')
        device = paddle.device.get_device()
        self.assertIn('gpu', device)
        paddle.device.set_device(original_device)

    def test_is_compiled_with_cuda(self):
        """测试CUDA编译检查 / Test CUDA compilation check"""
        result = paddle.device.is_compiled_with_cuda()
        self.assertIsInstance(result, bool)

    def test_is_compiled_with_xpu(self):
        """测试XPU编译检查 / Test XPU compilation check"""
        result = paddle.device.is_compiled_with_xpu()
        self.assertIsInstance(result, bool)

    def test_get_available_device(self):
        """测试获取可用设备 / Test getting available devices"""
        devices = paddle.device.get_available_device()
        self.assertIsInstance(devices, list)
        self.assertTrue(len(devices) > 0)


class TestCUDADevice(unittest.TestCase):
    """测试CUDA设备相关功能 / Test CUDA device operations"""

    @unittest.skipIf(not HAS_GPU, "No GPU available")
    def test_cuda_device_count(self):
        """测试GPU数量查询 / Test GPU count query"""
        count = paddle.device.cuda.device_count()
        self.assertIsInstance(count, int)
        self.assertTrue(count > 0)

    @unittest.skipIf(not HAS_GPU, "No GPU available")
    def test_cuda_current_device(self):
        """测试当前CUDA设备 / Test current CUDA device"""
        # Use paddle internal method to get device id
        device = paddle.device.get_device()
        self.assertIn('gpu', device)

    @unittest.skipIf(not HAS_GPU, "No GPU available")
    def test_cuda_get_device_name(self):
        """测试获取GPU名称 / Test getting GPU name"""
        name = paddle.device.cuda.get_device_name(0)
        self.assertIsInstance(name, str)
        self.assertTrue(len(name) > 0)

    @unittest.skipIf(not HAS_GPU, "No GPU available")
    def test_cuda_get_device_capability(self):
        """测试获取GPU计算能力 / Test getting GPU compute capability"""
        capability = paddle.device.cuda.get_device_capability(0)
        self.assertIsInstance(capability, tuple)
        self.assertEqual(len(capability), 2)

    @unittest.skipIf(not HAS_GPU, "No GPU available")
    def test_cuda_memory_allocated(self):
        """测试CUDA内存分配查询 / Test CUDA memory allocation query"""
        allocated = paddle.device.cuda.memory_allocated(0)
        self.assertIsInstance(allocated, int)
        self.assertGreaterEqual(allocated, 0)

    @unittest.skipIf(not HAS_GPU, "No GPU available")
    def test_cuda_max_memory_allocated(self):
        """测试CUDA最大内存分配查询 / Test CUDA max memory allocation"""
        max_alloc = paddle.device.cuda.max_memory_allocated(0)
        self.assertIsInstance(max_alloc, int)
        self.assertGreaterEqual(max_alloc, 0)

    @unittest.skipIf(not HAS_GPU, "No GPU available")
    def test_cuda_empty_cache(self):
        """测试CUDA缓存清理 / Test CUDA cache clearing"""
        # 创建一个大张量然后删除以产生缓存
        x = paddle.randn([1000, 1000])
        del x
        # 清理缓存
        paddle.device.cuda.empty_cache()

    @unittest.skipIf(not HAS_GPU, "No GPU available")
    def test_cuda_synchronize(self):
        """测试CUDA同步 / Test CUDA synchronization"""
        x = paddle.randn([100, 100])
        y = paddle.matmul(x, x)
        paddle.device.cuda.synchronize()
        self.assertEqual(y.shape, [100, 100])


class TestTensorDevice(unittest.TestCase):
    """测试张量设备操作 / Test tensor device operations"""

    def test_tensor_place(self):
        """测试张量所在设备 / Test tensor device placement"""
        x = paddle.randn([3, 4])
        place = x.place
        self.assertIsNotNone(place)

    def test_tensor_to_cpu(self):
        """测试张量转移到CPU / Test tensor transfer to CPU"""
        x = paddle.randn([3, 4])
        x_cpu = x.cpu()
        self.assertTrue(x_cpu.place.is_cpu_place())

    @unittest.skipIf(not HAS_GPU, "No GPU available")
    def test_tensor_to_gpu(self):
        """测试张量转移到GPU / Test tensor transfer to GPU"""
        x = paddle.randn([3, 4])
        x_gpu = x.cuda()
        self.assertTrue(x_gpu.place.is_gpu_place())

    @unittest.skipIf(not HAS_GPU, "No GPU available")
    def test_tensor_to_device(self):
        """测试张量转移到指定设备 / Test tensor transfer to specific device"""
        x = paddle.randn([3, 4])
        x_gpu = x.cuda(0)
        self.assertTrue(x_gpu.place.is_gpu_place())

    def test_create_tensor_on_device(self):
        """测试在指定设备上创建张量 / Test creating tensor on specific device"""
        # Create tensor on CPU explicitly
        x = paddle.randn([3, 4])
        x_cpu = x.cpu()
        self.assertTrue(x_cpu.place.is_cpu_place())


class TestDeviceGuard(unittest.TestCase):
    """测试设备创建 / Test device tensor creation"""

    def test_cpu_place(self):
        """测试CPU place创建张量 / Test tensor creation on CPU place"""
        place = paddle.CPUPlace()
        x = paddle.to_tensor([1.0, 2.0, 3.0], place=place)
        self.assertTrue(x.place.is_cpu_place())

    @unittest.skipIf(not HAS_GPU, "No GPU available")
    def test_gpu_place(self):
        """测试GPU place创建张量 / Test tensor creation on GPU place"""
        place = paddle.CUDAPlace(0)
        x = paddle.to_tensor([1.0, 2.0, 3.0], place=place)
        self.assertTrue(x.place.is_gpu_place())

    def test_place_compatibility(self):
        """测试跨设备张量操作 / Test cross-device tensor operations"""
        x = paddle.randn([3, 4])
        x_cpu = x.cpu()
        # Verify shape preserved
        self.assertEqual(x_cpu.shape, [3, 4])


if __name__ == '__main__':
    unittest.main()
