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


class DeviceType:
    CPU = 'cpu'
    GPU = 'gpu'
    XPU = 'xpu'
    NPU = 'npu'
    MLU = 'mlu'


class Device(object):
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
        if self._dtype == DeviceType.NPU:
            return 'FLAGS_selected_npus'
        if self._dtype == DeviceType.XPU:
            return 'FLAGS_selected_xpus'
        if self._dtype == DeviceType.MLU:
            return 'FLAGS_selected_mlus'
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

    @classmethod
    def parse_device(self):
        dev = Device()
        visible_devices = None
        if 'CUDA_VISIBLE_DEVICES' in os.environ or 'NVIDIA_VISIBLE_DEVICES' in os.environ:
            dev._dtype = DeviceType.GPU
            visible_devices = os.getenv("CUDA_VISIBLE_DEVICES") or os.getenv(
                "NVIDIA_VISIBLE_DEVICES")
        elif 'XPU_VISIBLE_DEVICES' in os.environ:
            dev._dtype = DeviceType.XPU
            visible_devices = os.getenv("XPU_VISIBLE_DEVICES")
        elif 'ASCEND_VISIBLE_DEVICES' in os.environ:
            dev._dtype = DeviceType.NPU
            visible_devices = os.getenv("ASCEND_VISIBLE_DEVICES")
        elif 'MLU_VISIBLE_DEVICES' in os.environ:
            dev._dtype = DeviceType.MLU
            visible_devices = os.getenv("MLU_VISIBLE_DEVICES")

        if visible_devices is not None and visible_devices != 'all':
            dev._labels = visible_devices.split(',')
        else:
            return self.detect_device()

        return dev

    @classmethod
    def detect_device(self):
        import paddle.fluid as fluid

        dev = Device()
        num = 0
        visible_devices = None
        if fluid.core.is_compiled_with_cuda():
            dev._dtype = DeviceType.GPU
            num = fluid.core.get_cuda_device_count()
            visible_devices = os.getenv("CUDA_VISIBLE_DEVICES") or os.getenv(
                "NVIDIA_VISIBLE_DEVICES")
        elif fluid.core.is_compiled_with_xpu():
            dev._dtype = DeviceType.XPU
            num = fluid.core.get_xpu_device_count()
            visible_devices = os.getenv("XPU_VISIBLE_DEVICES")
        elif fluid.core.is_compiled_with_npu():
            dev._dtype = DeviceType.NPU
            num = fluid.core.get_npu_device_count()
            visible_devices = os.getenv("ASCEND_VISIBLE_DEVICES")
        elif fluid.core.is_compiled_with_mlu():
            dev._dtype = DeviceType.MLU
            num = fluid.core.get_mlu_device_count()
            visible_devices = os.getenv("MLU_VISIBLE_DEVICES")

        if num == 0:
            dev._dtype = DeviceType.CPU
        elif visible_devices is None or visible_devices == "all":
            dev._labels = [str(x) for x in range(0, num)]
        else:
            dev._labels = visible_devices.split(',')

        return dev


if __name__ == '__main__':
    d = Device.parse_device()
    print(d.get_selected_flag())
