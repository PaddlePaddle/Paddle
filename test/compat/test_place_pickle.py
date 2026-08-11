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

from __future__ import annotations

import copy
import pickle
import unittest

import paddle
from paddle.base import core


def _device_count(device_name: str) -> int:
    if device_name in ("gpu", "cuda"):
        return paddle.device.cuda.device_count()
    if device_name == "xpu":
        return paddle.device.xpu.device_count()
    if device_name == "ipu":
        return core.get_ipu_device_count()
    return 0


class TestPlacePickle(unittest.TestCase):
    def _assert_roundtrip(self, place):
        copies = [
            copy.copy(place),
            copy.deepcopy(place),
            pickle.loads(pickle.dumps(place)),
        ]
        for restored in copies:
            self.assertIs(type(restored), type(place))
            self.assertEqual(restored, place)
            self.assertEqual(str(restored), str(place))

    def test_cpu_place(self):
        self._assert_roundtrip(paddle.CPUPlace())

    @unittest.skipUnless(paddle.is_compiled_with_cuda(), "Requires CUDA build")
    def test_cuda_place(self):
        if _device_count("gpu") <= 0:
            self.skipTest("Requires a visible CUDA device")
        self._assert_roundtrip(paddle.CUDAPlace(0))
        self._assert_roundtrip(paddle.CUDAPinnedPlace())

    @unittest.skipUnless(
        paddle.device.is_compiled_with_xpu(), "Requires XPU build"
    )
    def test_xpu_place(self):
        if _device_count("xpu") <= 0:
            self.skipTest("Requires a visible XPU device")
        self._assert_roundtrip(paddle.XPUPlace(0))
        self._assert_roundtrip(paddle.XPUPinnedPlace())

    def test_custom_place(self):
        custom_device_types = [
            device_type
            for device_type in paddle.device.get_all_custom_device_type()
            if paddle.device.is_compiled_with_custom_device(device_type)
        ]
        if not custom_device_types:
            self.skipTest("Requires a visible custom device")
        for device_type in custom_device_types:
            try:
                place = paddle.CustomPlace(device_type, 0)
            except Exception:
                continue
            self._assert_roundtrip(place)
            return
        self.skipTest("Requires a visible custom device")

    @unittest.skipUnless(
        paddle.device.is_compiled_with_ipu(), "Requires IPU build"
    )
    def test_ipu_place(self):
        if _device_count("ipu") <= 0:
            self.skipTest("Requires a visible IPU device")
        self._assert_roundtrip(paddle.IPUPlace())


if __name__ == "__main__":
    unittest.main()
