# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

from paddle.device import get_available_custom_device

# (TODO: GhostScreaming) It will be removed later.
from paddle.fluid import core


class DeviceType:
    CPU = 'cpu'
    GPU = 'gpu'
    XPU = 'xpu'
    IPU = 'ipu'
    CUSTOM_DEVICE = 'custom_device'


class Device:
    def __init__(self, dtype=None, memory="", labels=""):
        self._dtype = dtype
        self._memory = memory
        self._labels = labels

    def __str__(self):
        return ",".join(self._labels)

    @property
    def dtype(self):
        return self._dtype

    @property
    def count(self):
        return len(self._labels) or 1

    @property
    def memory(self):
        return self._memory

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, lbs):
        if isinstance(lbs, str):
            self._labels = lbs.split(',')
        elif isinstance(lbs, list):
            self._labels = lbs
        else:
            self._labels = []

    def get_selected_device_key(self):
        if self._dtype == DeviceType.CPU:
            return 'FLAGS_selected_cpus'
        if self._dtype == DeviceType.GPU:
            return 'FLAGS_selected_gpus'
        if self._dtype == DeviceType.XPU:
            return 'FLAGS_selected_xpus'
        if self._dtype == DeviceType.IPU:
            return 'FLAGS_selected_ipus'
        if self._dtype == DeviceType.CUSTOM_DEVICE:
            return 'FLAGS_selected_{}s'.format(os.getenv('PADDLE_XCCL_BACKEND'))
        return 'FLAGS_selected_devices'

    def get_selected_devices(self, devices=''):
        '''
        return the device label/id relative to the visible devices
        '''
        if not devices:
            return [str(x) for x in range(0, len(self._labels))]
        else:
            devs = [x.strip() for x in devices.split(',')]
            return [str(self._labels.index(d)) for d in devs]

    def get_custom_device_envs(self):
        return {
            'PADDLE_DISTRI_BACKEND': 'xccl',
            'PADDLE_XCCL_BACKEND': os.getenv('PADDLE_XCCL_BACKEND'),
        }

    @classmethod
    def parse_device(self):
        dev = Device()
        visible_devices = None
        if 'PADDLE_XCCL_BACKEND' in os.environ:
            dev._dtype = DeviceType.CUSTOM_DEVICE
            visible_devices_str = '{}_VISIBLE_DEVICES'.format(
                os.getenv('PADDLE_XCCL_BACKEND').upper()
            )
            if visible_devices_str in os.environ:
                visible_devices = os.getenv(visible_devices_str)
        elif 'CUDA_VISIBLE_DEVICES' in os.environ:
            dev._dtype = DeviceType.GPU
            visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        elif 'XPU_VISIBLE_DEVICES' in os.environ:
            dev._dtype = DeviceType.XPU
            visible_devices = os.getenv("XPU_VISIBLE_DEVICES")

        if visible_devices is not None and visible_devices != 'all':
            dev._labels = visible_devices.split(',')
        else:
            return self.detect_device()

        return dev

    @classmethod
    def detect_device(self):
        def get_custom_devices_count(device_type):
            all_custom_devices = get_available_custom_device()
            all_custom_devices = [
                device.split(':')[0] for device in all_custom_devices
            ]
            custom_devices_count = all_custom_devices.count(device_type)
            return custom_devices_count

        dev = Device()
        num = 0
        visible_devices = None
        if 'PADDLE_XCCL_BACKEND' in os.environ:
            custom_device_type = os.getenv('PADDLE_XCCL_BACKEND')
            dev._dtype = DeviceType.CUSTOM_DEVICE
            num = get_custom_devices_count(custom_device_type)
            visible_devices_str = '{}_VISIBLE_DEVICES'.format(
                custom_device_type.upper()
            )
            if visible_devices_str in os.environ:
                visible_devices = os.getenv(visible_devices_str)
        elif core.is_compiled_with_cuda():
            dev._dtype = DeviceType.GPU
            num = core.get_cuda_device_count()
            visible_devices = os.getenv("CUDA_VISIBLE_DEVICES")
        elif core.is_compiled_with_xpu():
            dev._dtype = DeviceType.XPU
            num = core.get_xpu_device_count()
            visible_devices = os.getenv("XPU_VISIBLE_DEVICES")
        elif core.is_compiled_with_ipu():
            dev._dtype = DeviceType.IPU
            num = core.get_ipu_device_count()
            # For IPUs, 'labels' is a list which contains the available numbers of IPU devices.
            dev._labels = [str(x) for x in range(0, num + 1)]
            return dev

        if num == 0:
            dev._dtype = DeviceType.CPU
        elif visible_devices is None or visible_devices == "all":
            dev._labels = [str(x) for x in range(0, num)]
        else:
            dev._labels = visible_devices.split(',')

        return dev


if __name__ == '__main__':
    d = Device.parse_device()
    print(d.get_selected_devices())
