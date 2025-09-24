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

import paddle
from paddle import get_device_module


class TestGetDeviceModule(unittest.TestCase):
    def test_str_devices(self):
        self.assertIs(get_device_module("gpu:0"), paddle.cuda)
        self.assertIs(get_device_module("cuda:0"), paddle.cuda)

        self.assertIs(get_device_module("xpu:0"), paddle.device.xpu)

        custom_devices = [
            "metax_gpu",
            "biren_gpu",
            "custom_cpu",
            "gcu",
            "iluvatar_gpu",
            "intel_gpu",
            "intel_hpu",
            "mlu",
            "mps",
            "npu",
            "sdaa",
        ]
        for dev in custom_devices:
            self.assertIs(get_device_module(dev), paddle.device.custom_device)

        self.assertIs(get_device_module('cpu'), paddle.device)

        with self.assertRaises(RuntimeError):
            get_device_module("unknown_device")

    def test_place_devices(self):
        if paddle.cuda.is_available() and paddle.device.is_compiled_with_cuda():
            self.assertIs(get_device_module(paddle.CUDAPlace(0)), paddle.cuda)

    def test_none_device(self):
        current_device_module = get_device_module(None)
        current_device_type = paddle.device.get_device().split(":")[0].lower()
        if current_device_type in ("cuda", "gpu"):
            self.assertIs(current_device_module, paddle.cuda)
        elif current_device_type == "xpu":
            self.assertIs(current_device_module, paddle.device.xpu)
        elif current_device_type in [
            "metax_gpu",
            "biren_gpu",
            "custom_cpu",
            "gcu",
            "iluvatar_gpu",
            "intel_gpu",
            "intel_hpu",
            "mlu",
            "mps",
            "npu",
            "sdaa",
        ]:
            self.assertIs(current_device_module, paddle.device.custom_device)
        elif current_device_type == "cpu":
            self.assertIs(current_device_module, paddle.device)


if __name__ == "__main__":
    unittest.main()
