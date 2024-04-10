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

import numpy as np

import paddle


class TestTensorPlaceXPU(unittest.TestCase):
    def test_dynamic(self):
        paddle.disable_static()
        data = np.random.random(size=1024)
        x = paddle.to_tensor(data, place=paddle.CPUPlace())
        self.assertTrue(x.place.is_cpu_place())
        if paddle.device.xpu.device_count() > 0:
            y = x.xpu()
            self.assertTrue(y.place.is_xpu_place())
            self.assertEqual(y.place.xpu_device_id(), 0)
            y_data = y.numpy()
            np.testing.assert_equal(data, y_data)
        if paddle.device.xpu.device_count() > 1:
            y = x.xpu(1)
            self.assertTrue(y.place.is_xpu_place())
            self.assertEqual(y.place.xpu_device_id(), 1)
            y_data = y.numpy()
            np.testing.assert_equal(data, y_data)

    def test_to_static(self):
        def to_xpu(x):
            return x.xpu()

        paddle.disable_static()
        static_to_xpu = paddle.jit.to_static(to_xpu, full_graph=True)
        data = np.random.random(size=1024)
        x = paddle.to_tensor(data, place=paddle.CPUPlace())
        self.assertTrue(x.place.is_cpu_place())
        y = static_to_xpu(x)
        self.assertTrue(y.place.is_xpu_place())
        self.assertEqual(y.place.xpu_device_id(), 0)
        y_data = y.numpy()
        np.testing.assert_equal(data, y_data)


if __name__ == '__main__':
    unittest.main()
