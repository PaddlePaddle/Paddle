# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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


class TestTensorPlaceXPU(unittest.TestCase):
    def test_tensor(self):
        x = paddle.to_tensor(1.0, place=paddle.CPUPlace())
        self.assertTrue(x.place.is_cpu_place())
        if paddle.device.xpu.device_count() > 0:
            y = x.xpu()
            self.assertTrue(y.place.is_xpu_place())
            self.assertTrue(x.place.xpu_device_id, 0)
        if paddle.device.xpu.device_count() > 1:
            y = x.xpu(1)
            self.assertTrue(y.place.is_xpu_place())
            self.assertTrue(x.place.xpu_device_id, 1)


if __name__ == '__main__':
    paddle.disable_static()
    unittest.main()
