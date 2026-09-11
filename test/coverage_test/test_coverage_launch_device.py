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

# [AUTO-GENERATED] Test file for paddle.distributed.launch.context.device
# 覆盖模块: paddle/distributed/launch/context/device.py
# 未覆盖行: 38,46,50,58,59,60,61,63,67,70,71,72,73,74,75,76,79,80,87,93,94,95,96,97,98,106,107,108,109,112,113,114,115,116,117,118,119,121,122,124,125,127,129,134,135,138,141,142,143,144,145,146,147,148,149,150,152,153,154,155,156,157,158,159,160,161,162,163,164,166,167,169,170,171,172,173,174
# Covered module: paddle/distributed/launch/context/device.py
# Uncovered lines: 38,46,50,58,59,60,61,63,67,70,71,72,73,74,75,76,79,80,87,93,94,95,96,97,98

import unittest

from paddle.distributed.launch.context.device import Device, DeviceType


class TestDeviceType(unittest.TestCase):
    """测试 DeviceType 常量类
    Test DeviceType constant class"""

    def test_device_type_cpu(self):
        """测试 CPU 设备类型
        Test CPU device type"""
        self.assertEqual(DeviceType.CPU, 'cpu')

    def test_device_type_gpu(self):
        """测试 GPU 设备类型
        Test GPU device type"""
        self.assertEqual(DeviceType.GPU, 'gpu')

    def test_device_type_xpu(self):
        """测试 XPU 设备类型
        Test XPU device type"""
        self.assertEqual(DeviceType.XPU, 'xpu')

    def test_device_type_ipu(self):
        """测试 IPU 设备类型
        Test IPU device type"""
        self.assertEqual(DeviceType.IPU, 'ipu')

    def test_device_type_custom(self):
        """测试自定义设备类型
        Test custom device type"""
        self.assertEqual(DeviceType.CUSTOM_DEVICE, 'custom_device')


class TestDevice(unittest.TestCase):
    """测试 Device 类
    Test Device class"""

    def test_device_init_default(self):
        """测试 Device 默认初始化
        Test Device default initialization"""
        device = Device()
        self.assertIsNone(device.dtype)
        self.assertEqual(device.memory, "")
        self.assertEqual(device.labels, "")

    def test_device_init_with_params(self):
        """测试 Device 带参数初始化
        Test Device initialization with parameters"""
        device = Device(
            dtype=DeviceType.GPU, memory="32GB", labels=["0", "1", "2"]
        )
        self.assertEqual(device.dtype, DeviceType.GPU)
        self.assertEqual(device.memory, "32GB")
        self.assertEqual(device.labels, ["0", "1", "2"])

    def test_device_str(self):
        """测试 Device.__str__ 方法
        Test Device.__str__ method"""
        device = Device(dtype=DeviceType.GPU, labels=["0", "1", "2"])
        self.assertEqual(str(device), "0,1,2")

    def test_device_count_with_labels(self):
        """测试 Device.count 有标签时返回标签数量
        Test Device.count returns labels count when labels exist"""
        device = Device(dtype=DeviceType.GPU, labels=["0", "1", "2", "3"])
        self.assertEqual(device.count, 4)

    def test_device_count_no_labels(self):
        """测试 Device.count 无标签时返回1
        Test Device.count returns 1 when no labels"""
        device = Device(dtype=DeviceType.CPU)
        self.assertEqual(device.count, 1)

    def test_device_labels_setter_string(self):
        """测试 Device.labels setter 接受字符串
        Test Device.labels setter accepts string"""
        device = Device()
        device.labels = "0,1,2"
        self.assertEqual(device.labels, ["0", "1", "2"])

    def test_device_labels_setter_list(self):
        """测试 Device.labels setter 接受列表
        Test Device.labels setter accepts list"""
        device = Device()
        device.labels = ["0", "1", "2"]
        self.assertEqual(device.labels, ["0", "1", "2"])

    def test_device_labels_setter_other(self):
        """测试 Device.labels setter 接受其他类型时设置为空
        Test Device.labels setter sets empty for other types"""
        device = Device()
        device.labels = 123
        self.assertEqual(device.labels, [])

    def test_device_get_selected_device_key_cpu(self):
        """测试 CPU 设备的 selected_device_key
        Test CPU device selected_device_key"""
        device = Device(dtype=DeviceType.CPU)
        self.assertEqual(
            device.get_selected_device_key(), 'FLAGS_selected_cpus'
        )

    def test_device_get_selected_device_key_gpu(self):
        """测试 GPU 设备的 selected_device_key
        Test GPU device selected_device_key"""
        device = Device(dtype=DeviceType.GPU)
        self.assertEqual(
            device.get_selected_device_key(), 'FLAGS_selected_gpus'
        )

    def test_device_get_selected_device_key_xpu(self):
        """测试 XPU 设备的 selected_device_key
        Test XPU device selected_device_key"""
        device = Device(dtype=DeviceType.XPU)
        self.assertEqual(
            device.get_selected_device_key(), 'FLAGS_selected_xpus'
        )

    def test_device_get_selected_device_key_ipu(self):
        """测试 IPU 设备的 selected_device_key
        Test IPU device selected_device_key"""
        device = Device(dtype=DeviceType.IPU)
        self.assertEqual(
            device.get_selected_device_key(), 'FLAGS_selected_ipus'
        )

    def test_device_get_selected_device_key_unknown(self):
        """测试未知设备的 selected_device_key
        Test unknown device selected_device_key"""
        device = Device(dtype="unknown")
        self.assertEqual(
            device.get_selected_device_key(), 'FLAGS_selected_devices'
        )

    def test_device_get_selected_devices_empty(self):
        """测试无可见设备时的 get_selected_devices
        Test get_selected_devices with no visible devices"""
        device = Device(dtype=DeviceType.GPU, labels=["0", "1", "2"])
        result = device.get_selected_devices()
        self.assertEqual(result, ["0", "1", "2"])

    def test_device_get_selected_devices_with_spec(self):
        """测试指定设备的 get_selected_devices
        Test get_selected_devices with specified devices"""
        device = Device(dtype=DeviceType.GPU, labels=["0", "1", "2"])
        result = device.get_selected_devices("0,2")
        self.assertEqual(result, ["0", "2"])

    def test_device_memory_property(self):
        """测试 Device.memory 属性
        Test Device.memory property"""
        device = Device(memory="16384MB")
        self.assertEqual(device.memory, "16384MB")


if __name__ == '__main__':
    unittest.main()
