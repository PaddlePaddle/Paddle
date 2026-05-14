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
# Target file: paddle/distributed/launch/context/device.py
# 覆盖模块: paddle/distributed/launch/context/device.py
# Covered module: paddle/distributed/launch/context/device.py

import os
import sys
import unittest
from unittest.mock import patch

from paddle.distributed.launch.context.device import Device, DeviceType

# Get module reference for patching (dotted path doesn't work for launch module)
_device_mod = sys.modules['paddle.distributed.launch.context.device']


class TestDeviceType(unittest.TestCase):
    """测试 DeviceType 常量类 / Test DeviceType constant class"""

    def test_device_type_cpu(self):
        """测试 CPU 设备类型 / Test CPU device type"""
        self.assertEqual(DeviceType.CPU, 'cpu')

    def test_device_type_gpu(self):
        """测试 GPU 设备类型 / Test GPU device type"""
        self.assertEqual(DeviceType.GPU, 'gpu')

    def test_device_type_xpu(self):
        """测试 XPU 设备类型 / Test XPU device type"""
        self.assertEqual(DeviceType.XPU, 'xpu')

    def test_device_type_ipu(self):
        """测试 IPU 设备类型 / Test IPU device type"""
        self.assertEqual(DeviceType.IPU, 'ipu')

    def test_device_type_custom(self):
        """测试自定义设备类型 / Test custom device type"""
        self.assertEqual(DeviceType.CUSTOM_DEVICE, 'custom_device')


class TestDevice(unittest.TestCase):
    """测试 Device 类 / Test Device class"""

    def test_device_init_default(self):
        """测试 Device 默认初始化 / Test Device default initialization"""
        device = Device()
        self.assertIsNone(device.dtype)
        self.assertEqual(device.memory, "")
        self.assertEqual(device.labels, "")

    def test_device_init_with_params(self):
        """测试 Device 带参数初始化 / Test Device initialization with parameters"""
        device = Device(
            dtype=DeviceType.GPU, memory="32GB", labels=["0", "1", "2"]
        )
        self.assertEqual(device.dtype, DeviceType.GPU)
        self.assertEqual(device.memory, "32GB")
        self.assertEqual(device.labels, ["0", "1", "2"])

    def test_device_str_empty_labels(self):
        """测试空标签的 Device.__str__ / Test Device.__str__ with empty labels"""
        device = Device(dtype=DeviceType.CPU)
        self.assertEqual(str(device), "")

    def test_device_str_with_labels(self):
        """测试 Device.__str__ 方法 / Test Device.__str__ method"""
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

    def test_device_labels_setter_tuple(self):
        """测试 Device.labels setter 接受元组时设置为空
        Test Device.labels setter sets empty for tuple"""
        device = Device()
        device.labels = ("0", "1")
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

    @patch.object(_device_mod, 'get_all_custom_device_type', return_value=[])
    def test_device_get_selected_device_key_custom_empty(self, mock_custom):
        """测试自定义设备无设备类型时的 selected_device_key
        Test custom device with no device types selected_device_key"""
        device = Device(dtype=DeviceType.CUSTOM_DEVICE)
        key = device.get_selected_device_key()
        self.assertEqual(key, 'FLAGS_selected_s')

    @patch.object(
        _device_mod, 'get_all_custom_device_type', return_value=['npu']
    )
    def test_device_get_selected_device_key_custom_with_type(self, mock_custom):
        """测试自定义设备有设备类型时的 selected_device_key
        Test custom device with device type selected_device_key"""
        device = Device(dtype=DeviceType.CUSTOM_DEVICE)
        key = device.get_selected_device_key()
        self.assertEqual(key, 'FLAGS_selected_npus')

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

    def test_device_get_selected_devices_single(self):
        """测试单个指定设备的 get_selected_devices
        Test get_selected_devices with single specified device"""
        device = Device(dtype=DeviceType.GPU, labels=["0", "1", "2"])
        result = device.get_selected_devices("1")
        self.assertEqual(result, ["1"])

    def test_device_memory_property(self):
        """测试 Device.memory 属性 / Test Device.memory property"""
        device = Device(memory="16384MB")
        self.assertEqual(device.memory, "16384MB")

    def test_device_dtype_property(self):
        """测试 Device.dtype 属性 / Test Device.dtype property"""
        device = Device(dtype=DeviceType.GPU)
        self.assertEqual(device.dtype, DeviceType.GPU)


class TestDeviceCustomDeviceEnvs(unittest.TestCase):
    """测试 Device.get_custom_device_envs / Test Device.get_custom_device_envs"""

    @patch.object(
        _device_mod, 'get_all_custom_device_type', return_value=['npu']
    )
    def test_get_custom_device_envs_with_type(self, mock_custom):
        """测试有自定义设备类型时的环境变量 / Test envs with custom device type"""
        device = Device(dtype=DeviceType.CUSTOM_DEVICE)
        envs = device.get_custom_device_envs()
        self.assertEqual(envs['PADDLE_DISTRI_BACKEND'], 'xccl')
        self.assertEqual(envs['PADDLE_XCCL_BACKEND'], 'npu')

    @patch.object(_device_mod, 'get_all_custom_device_type', return_value=[])
    def test_get_custom_device_envs_no_type(self, mock_custom):
        """测试无自定义设备类型时的环境变量 / Test envs without custom device type"""
        device = Device(dtype=DeviceType.CUSTOM_DEVICE)
        envs = device.get_custom_device_envs()
        self.assertEqual(envs['PADDLE_DISTRI_BACKEND'], 'xccl')
        self.assertEqual(envs['PADDLE_XCCL_BACKEND'], '')


class TestDeviceParseDevice(unittest.TestCase):
    """测试 Device.parse_device 类方法 / Test Device.parse_device class method"""

    @patch.object(_device_mod, 'get_all_custom_device_type', return_value=[])
    @patch.object(_device_mod.core, 'is_compiled_with_xpu', return_value=False)
    def test_parse_device_cuda_visible(self, mock_xpu, mock_custom):
        """测试解析 CUDA 可见设备 / Test parsing CUDA visible devices"""
        env_backup = {}
        env_clear = [
            'XPULINK_VISIBLE_DEVICES',
            'XPU_VISIBLE_DEVICES',
            'NPU_VISIBLE_DEVICES',
        ]
        for k in env_clear:
            if k in os.environ:
                env_backup[k] = os.environ[k]
                del os.environ[k]
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
        try:
            dev = Device.parse_device()
            self.assertEqual(dev.dtype, DeviceType.GPU)
            self.assertEqual(dev.labels, ["0", "1", "2"])
        finally:
            del os.environ['CUDA_VISIBLE_DEVICES']
            for k, v in env_backup.items():
                os.environ[k] = v

    @patch.object(_device_mod, 'get_all_custom_device_type', return_value=[])
    def test_parse_device_xpu_visible(self, mock_custom):
        """测试解析 XPU 可见设备 / Test parsing XPU visible devices"""
        env_backup = {}
        env_clear = [
            'XPULINK_VISIBLE_DEVICES',
            'CUDA_VISIBLE_DEVICES',
            'NPU_VISIBLE_DEVICES',
        ]
        for k in env_clear:
            if k in os.environ:
                env_backup[k] = os.environ[k]
                del os.environ[k]
        os.environ['XPU_VISIBLE_DEVICES'] = '0,1'
        try:
            dev = Device.parse_device()
            self.assertEqual(dev.dtype, DeviceType.XPU)
            self.assertEqual(dev.labels, ["0", "1"])
        finally:
            del os.environ['XPU_VISIBLE_DEVICES']
            for k, v in env_backup.items():
                os.environ[k] = v

    @patch.object(_device_mod, 'get_all_custom_device_type', return_value=[])
    def test_parse_device_xpulink_visible(self, mock_custom):
        """测试解析 XPULINK 可见设备 / Test parsing XPULINK visible devices"""
        env_backup = {}
        env_clear = [
            'XPU_VISIBLE_DEVICES',
            'CUDA_VISIBLE_DEVICES',
            'NPU_VISIBLE_DEVICES',
        ]
        for k in env_clear:
            if k in os.environ:
                env_backup[k] = os.environ[k]
                del os.environ[k]
        os.environ['XPULINK_VISIBLE_DEVICES'] = '0,1,2,3'
        try:
            dev = Device.parse_device()
            self.assertEqual(dev.dtype, DeviceType.XPU)
            self.assertEqual(dev.labels, ["0", "1", "2", "3"])
        finally:
            del os.environ['XPULINK_VISIBLE_DEVICES']
            for k, v in env_backup.items():
                os.environ[k] = v

    @patch.object(
        _device_mod, 'get_all_custom_device_type', return_value=['custom_dev']
    )
    def test_parse_device_custom_device(self, mock_custom):
        """测试解析自定义设备 / Test parsing custom device"""
        env_backup = {}
        env_clear = [
            'XPULINK_VISIBLE_DEVICES',
            'XPU_VISIBLE_DEVICES',
            'CUDA_VISIBLE_DEVICES',
            'NPU_VISIBLE_DEVICES',
            'CUSTOM_DEVICE_VISIBLE_DEVICES',
            'CUSTOM_DEV_VISIBLE_DEVICES',
        ]
        for k in env_clear:
            if k in os.environ:
                env_backup[k] = os.environ[k]
                del os.environ[k]
        # Source code constructs: f'{device_type.upper()}_VISIBLE_DEVICES'
        # so for 'custom_dev' it looks for 'CUSTOM_DEV_VISIBLE_DEVICES'
        os.environ['CUSTOM_DEV_VISIBLE_DEVICES'] = '0,1'
        try:
            dev = Device.parse_device()
            self.assertEqual(dev.dtype, DeviceType.CUSTOM_DEVICE)
            self.assertEqual(dev.labels, ["0", "1"])
        finally:
            del os.environ['CUSTOM_DEV_VISIBLE_DEVICES']
            for k, v in env_backup.items():
                os.environ[k] = v

    @patch.object(_device_mod, 'get_all_custom_device_type', return_value=[])
    @patch.object(_device_mod.core, 'is_compiled_with_xpu', return_value=False)
    def test_parse_device_cuda_visible_all(self, mock_xpu, mock_custom):
        """测试解析 CUDA_VISIBLE=all 时走 detect_device
        Test parsing CUDA_VISIBLE=all goes to detect_device"""
        env_backup = {}
        env_clear = [
            'XPULINK_VISIBLE_DEVICES',
            'XPU_VISIBLE_DEVICES',
            'NPU_VISIBLE_DEVICES',
        ]
        for k in env_clear:
            if k in os.environ:
                env_backup[k] = os.environ[k]
                del os.environ[k]
        os.environ['CUDA_VISIBLE_DEVICES'] = 'all'
        try:
            with patch.object(Device, 'detect_device') as mock_detect:
                mock_detect.return_value = Device()
                Device.parse_device()
                mock_detect.assert_called_once()
        finally:
            del os.environ['CUDA_VISIBLE_DEVICES']
            for k, v in env_backup.items():
                os.environ[k] = v


class TestDeviceDetectDevice(unittest.TestCase):
    """测试 Device.detect_device 类方法 / Test Device.detect_device class method"""

    @patch.object(_device_mod, 'get_all_custom_device_type', return_value=[])
    @patch.object(_device_mod.core, 'is_compiled_with_cuda', return_value=True)
    @patch.object(_device_mod.core, 'get_cuda_device_count', return_value=4)
    def test_detect_device_cuda(self, mock_count, mock_cuda, mock_custom):
        """测试检测 CUDA 设备 / Test detecting CUDA device"""
        env_backup = os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        env_backup2 = os.environ.pop('XPU_VISIBLE_DEVICES', None)
        try:
            dev = Device.detect_device()
            self.assertEqual(dev.dtype, DeviceType.GPU)
            self.assertEqual(dev.labels, ["0", "1", "2", "3"])
        finally:
            if env_backup is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = env_backup
            if env_backup2 is not None:
                os.environ['XPU_VISIBLE_DEVICES'] = env_backup2

    @patch.object(_device_mod, 'get_all_custom_device_type', return_value=[])
    @patch.object(_device_mod.core, 'is_compiled_with_cuda', return_value=False)
    @patch.object(_device_mod.core, 'is_compiled_with_xpu', return_value=True)
    @patch.object(
        _device_mod.core, 'get_xpu_device_count', return_value=2, create=True
    )
    def test_detect_device_xpu(
        self, mock_count, mock_xpu, mock_cuda, mock_custom
    ):
        """测试检测 XPU 设备 / Test detecting XPU device"""
        env_backup = os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        env_backup2 = os.environ.pop('XPU_VISIBLE_DEVICES', None)
        try:
            dev = Device.detect_device()
            self.assertEqual(dev.dtype, DeviceType.XPU)
            self.assertEqual(dev.labels, ["0", "1"])
        finally:
            if env_backup is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = env_backup
            if env_backup2 is not None:
                os.environ['XPU_VISIBLE_DEVICES'] = env_backup2

    @patch.object(_device_mod, 'get_all_custom_device_type', return_value=[])
    @patch.object(_device_mod.core, 'is_compiled_with_cuda', return_value=False)
    @patch.object(_device_mod.core, 'is_compiled_with_xpu', return_value=False)
    @patch.object(_device_mod.core, 'is_compiled_with_ipu', return_value=True)
    @patch.object(
        _device_mod.core, 'get_ipu_device_count', return_value=4, create=True
    )
    def test_detect_device_ipu(
        self, mock_count, mock_ipu, mock_xpu, mock_cuda, mock_custom
    ):
        """测试检测 IPU 设备 / Test detecting IPU device"""
        env_backup = os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        try:
            dev = Device.detect_device()
            self.assertEqual(dev.dtype, DeviceType.IPU)
            # IPU labels include num + 1
            self.assertEqual(len(dev.labels), 5)
        finally:
            if env_backup is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = env_backup

    @patch.object(_device_mod, 'get_all_custom_device_type', return_value=[])
    @patch.object(_device_mod.core, 'is_compiled_with_cuda', return_value=False)
    @patch.object(_device_mod.core, 'is_compiled_with_xpu', return_value=False)
    @patch.object(_device_mod.core, 'is_compiled_with_ipu', return_value=False)
    def test_detect_device_cpu_fallback(
        self, mock_ipu, mock_xpu, mock_cuda, mock_custom
    ):
        """测试检测无 GPU/XPU 时回退到 CPU
        Test detecting fallback to CPU when no GPU/XPU"""
        env_backup = os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        env_backup2 = os.environ.pop('XPU_VISIBLE_DEVICES', None)
        try:
            dev = Device.detect_device()
            self.assertEqual(dev.dtype, DeviceType.CPU)
        finally:
            if env_backup is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = env_backup
            if env_backup2 is not None:
                os.environ['XPU_VISIBLE_DEVICES'] = env_backup2

    @patch.object(_device_mod, 'get_all_custom_device_type', return_value=[])
    @patch.object(_device_mod.core, 'is_compiled_with_cuda', return_value=True)
    @patch.object(_device_mod.core, 'get_cuda_device_count', return_value=0)
    def test_detect_device_zero_count(self, mock_count, mock_cuda, mock_custom):
        """检测设备数量为0回退到 CPU / Test detecting zero devices falls back to CPU"""
        env_backup = os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        env_backup2 = os.environ.pop('XPU_VISIBLE_DEVICES', None)
        try:
            dev = Device.detect_device()
            self.assertEqual(dev.dtype, DeviceType.CPU)
        finally:
            if env_backup is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = env_backup
            if env_backup2 is not None:
                os.environ['XPU_VISIBLE_DEVICES'] = env_backup2

    @patch.object(_device_mod, 'get_all_custom_device_type', return_value=[])
    @patch.object(_device_mod.core, 'is_compiled_with_cuda', return_value=True)
    @patch.object(_device_mod.core, 'get_cuda_device_count', return_value=8)
    def test_detect_device_cuda_visible_subset(
        self, mock_count, mock_cuda, mock_custom
    ):
        """测试 CUDA 可见设备子集 / Test CUDA visible devices subset"""
        env_backup = os.environ.pop('CUDA_VISIBLE_DEVICES', None)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,2,4,6'
        try:
            dev = Device.detect_device()
            self.assertEqual(dev.dtype, DeviceType.GPU)
            self.assertEqual(dev.labels, ["0", "2", "4", "6"])
        finally:
            if env_backup is not None:
                os.environ['CUDA_VISIBLE_DEVICES'] = env_backup
            else:
                del os.environ['CUDA_VISIBLE_DEVICES']

    @patch.object(
        _device_mod, 'get_all_custom_device_type', return_value=['npu']
    )
    @patch.object(
        _device_mod,
        'get_available_custom_device',
        return_value=['npu:0', 'npu:1', 'npu:2'],
    )
    def test_detect_device_custom_device(self, mock_avail, mock_custom):
        """测试检测自定义设备 / Test detecting custom device"""
        env_backup = os.environ.pop('NPU_VISIBLE_DEVICES', None)
        try:
            dev = Device.detect_device()
            self.assertEqual(dev.dtype, DeviceType.CUSTOM_DEVICE)
            self.assertEqual(dev.labels, ["0", "1", "2"])
        finally:
            if env_backup is not None:
                os.environ['NPU_VISIBLE_DEVICES'] = env_backup


if __name__ == '__main__':
    unittest.main()
