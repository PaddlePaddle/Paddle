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


class Device(object):
    def __init__(self, dtype=None, count=1, memory="", labels=""):
        self.dtype = dtype
        self.count = count
        self.memory = memory
        self.labels = labels

    def labels_string(self):
        return ",".join(self.labels)

    @classmethod
    def parse_device(self):
        dev = Device()
        visible_devices = None
        if 'CUDA_VISIBLE_DEVICES' in os.environ or 'NVIDIA_VISIBLE_DEVICES' in os.environ:
            dev.dtype = DeviceType.GPU
            visible_devices = os.getenv("CUDA_VISIBLE_DEVICES") or os.getenv(
                "NVIDIA_VISIBLE_DEVICES")
        elif 'XPU_VISIBLE_DEVICES' in os.environ:
            dev.dtype = DeviceType.XPU
            visible_devices = os.getenv("XPU_VISIBLE_DEVICES")
        elif 'ASCEND_VISIBLE_DEVICES' in os.environ:
            dev.dtype = DeviceType.NPU
            visible_devices = os.getenv("ASCEND_VISIBLE_DEVICES")

        if visible_devices and visible_devices != 'all':
            dev.labels = visible_devices.split(',')
            dev.count = len(dev.labels)
        else:
            return self.detect_device()

        return dev

    @classmethod
    def detect_device(self):
        # enable paddle detection is very expansive
        import paddle.fluid as fluid

        dev = Device()
        num = 0
        visible_devices = None
        if fluid.core.is_compiled_with_cuda():
            dev.dtype = DeviceType.GPU
            num = fluid.core.get_cuda_device_count()
            visible_devices = os.getenv("CUDA_VISIBLE_DEVICES") or os.getenv(
                "NVIDIA_VISIBLE_DEVICES")
        elif fluid.core.is_compiled_with_xpu():
            dev.dtype = DeviceType.XPU
            num = fluid.core.get_xpu_device_count()
            visible_devices = os.getenv("XPU_VISIBLE_DEVICES")
        elif fluid.core.is_compiled_with_npu():
            dev.dtype = DeviceType.NPU
            num = fluid.core.get_npu_device_count()
            visible_devices = os.getenv("ASCEND_VISIBLE_DEVICES")

        if num == 0:
            dev.dtype = DeviceType.CPU
        elif visible_devices is None or visible_devices == "all" or visible_devices == "":
            dev.labels = [str(x) for x in range(0, num)]
            dev.count = num
        else:
            dev.labels = visible_devices.split(',')
            dev.count = len(dev.labels)

        return dev
