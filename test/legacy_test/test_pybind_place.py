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


class TestPybindPlace(unittest.TestCase):
    def test_cpu_place(self):
        pybind_place = paddle.CPUPlace()
        self.assertEqual(pybind_place, pybind_place)
        tensor_place = paddle.randn([2, 2]).to(device="cpu").place
        self.assertEqual(pybind_place, tensor_place)
        self.assertEqual(tensor_place, pybind_place)
        self.assertEqual(tensor_place, tensor_place)

        tensor_place_2 = paddle.randn([2, 2]).to(device="cpu").place
        self.assertEqual(tensor_place_2, tensor_place)
        self.assertEqual(tensor_place, tensor_place_2)

        pybind_place_2 = paddle.CPUPlace()
        self.assertEqual(pybind_place, pybind_place_2)

    def test_cuda_place(self):
        if paddle.device.is_compiled_with_cuda():
            pybind_place = paddle.CUDAPlace(0)
            self.assertEqual(pybind_place, pybind_place)
            tensor_place = paddle.randn([2, 2]).place
            self.assertEqual(pybind_place, tensor_place)
            self.assertEqual(tensor_place, pybind_place)
            self.assertEqual(tensor_place, tensor_place)

            tensor_place_2 = paddle.randn([2, 2]).place
            self.assertEqual(tensor_place_2, tensor_place)
            self.assertEqual(tensor_place, tensor_place_2)

            pybind_place_2 = paddle.CUDAPlace(0)
            self.assertEqual(pybind_place, pybind_place_2)
        else:
            self.skipTest("Skip as paddle is not compiled with cuda")

    def test_xpu_place(self):
        if paddle.device.is_compiled_with_xpu():
            pybind_place = paddle.XPUPlace(0)
            self.assertEqual(pybind_place, pybind_place)
            tensor_place = paddle.randn([2, 2]).place
            self.assertEqual(pybind_place, tensor_place)
            self.assertEqual(tensor_place, pybind_place)
            self.assertEqual(tensor_place, tensor_place)

            tensor_place_2 = paddle.randn([2, 2]).place
            self.assertEqual(tensor_place_2, tensor_place)
            self.assertEqual(tensor_place, tensor_place_2)

            pybind_place_2 = paddle.XPUPlace(0)
            self.assertEqual(pybind_place, pybind_place_2)
        else:
            self.skipTest("Skip as paddle is not compiled with xpu")

    def test_custom_place(self):
        if paddle.device.is_compiled_with_custom_device("FakeCPU"):
            pybind_place = paddle.CustomPlace("FakeCPU", 0)
            self.assertEqual(pybind_place, pybind_place)
            tensor_place = paddle.randn([2, 2]).place
            self.assertEqual(pybind_place, tensor_place)
            self.assertEqual(tensor_place, pybind_place)
            self.assertEqual(tensor_place, tensor_place)

            tensor_place_2 = paddle.randn([2, 2]).place
            self.assertEqual(tensor_place_2, tensor_place)
            self.assertEqual(tensor_place, tensor_place_2)

            pybind_place_2 = paddle.CustomPlace("FakeCPU", 0)
            self.assertEqual(pybind_place, pybind_place_2)
        else:
            self.skipTest("Skip as paddle is not compiled with custom device")

    def test_ipu_place(self):
        if paddle.device.is_compiled_with_ipu():
            pybind_place = paddle.IPUPlace()
            self.assertEqual(pybind_place, pybind_place)
            tensor_place = paddle.randn([2, 2]).place
            self.assertEqual(pybind_place, tensor_place)
            self.assertEqual(tensor_place, pybind_place)
            self.assertEqual(tensor_place, tensor_place)

            tensor_place_2 = paddle.randn([2, 2]).place
            self.assertEqual(tensor_place_2, tensor_place)
            self.assertEqual(tensor_place, tensor_place_2)

            pybind_place_2 = paddle.IPUPlace()
            self.assertEqual(pybind_place, pybind_place_2)
        else:
            self.skipTest("Skip as paddle is not compiled with ipu")


if __name__ == '__main__':
    unittest.main()
