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

from utils import dygraph_guard

import paddle


class TestPlaceGuard(unittest.TestCase):
    def test_str_place_obj_consistency(self):
        places = [
            ["cpu", paddle.CPUPlace()],
        ]
        if paddle.device.is_compiled_with_cuda():
            places.append(["gpu", paddle.CUDAPlace(0)])
            places.append(["gpu:0", paddle.CUDAPlace(0)])
        elif paddle.device.is_compiled_with_ipu():
            places.append(["ipu", paddle.IPUPlace()])
        elif paddle.device.is_compiled_with_xpu():
            places.append(["xpu:0", paddle.XPUPlace(0)])

        with dygraph_guard():
            for place_str, place_obj in places:
                with paddle.device.device_guard(place_str):
                    x = paddle.randn([2, 2])
                    x = x.tanh() ** 2
                    self.assertEqual(x.place, place_obj)

    def test_str_place_obj_scope_in_device(self):
        places = []
        if paddle.device.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
            places.append(paddle.CUDAPlace(0))
        elif paddle.device.is_compiled_with_ipu():
            places.append(paddle.IPUPlace())
        elif paddle.device.is_compiled_with_xpu():
            places.append(paddle.XPUPlace(0))
            places.append(paddle.XPUPlace(0))

        with dygraph_guard():
            for place_obj in places:
                x = paddle.randn([2, 2])  # create on default place
                with paddle.device.device_guard("cpu"):
                    x = (
                        x.tanh() ** 2
                    )  # should be still in place rather than cpu
                    self.assertNotEqual(x.place, paddle.CPUPlace())
                    self.assertEqual(x.place, place_obj)

    def test_wrong_device_name(self):
        with (
            dygraph_guard(),
            self.assertRaisesRegex(
                ValueError,
                "The device must be a string which is like 'cpu', 'gpu', 'gpu:x',",
            ),
            paddle.device.device_guard("xxx"),
        ):
            pass

    def test_wrong_device_type(self):
        with (
            dygraph_guard(),
            self.assertRaisesRegex(
                ValueError,
                "'device' must be a string or an instance of a subclass of",
            ),
            paddle.device.device_guard(paddle.randn([2])),
        ):
            pass

    def test_str_place_obj_nested(self):
        places = [paddle.CPUPlace()]
        if paddle.device.is_compiled_with_cuda():
            places.append(paddle.CUDAPlace(0))
            places.append(paddle.CUDAPlace(0))
        elif paddle.device.is_compiled_with_ipu():
            places.append(paddle.IPUPlace())
        elif paddle.device.is_compiled_with_xpu():
            places.append(paddle.XPUPlace(0))
            places.append(paddle.XPUPlace(0))

        if len(places) >= 2:
            place_obj1, place_obj2 = places[:2]
        else:
            self.skipTest("Not compiled with HPC hardware.")

        with dygraph_guard():
            with paddle.device.device_guard(place_obj1):
                x = paddle.randn([2, 2])  # create on place1
                self.assertEqual(x.place, place_obj1)
                self.assertNotEqual(x.place, place_obj2)

                with paddle.device.device_guard(place_obj2):
                    xx = paddle.randn([2, 2])  # create on place1
                    self.assertEqual(xx.place, place_obj2)
                    self.assertNotEqual(xx.place, place_obj1)

                    with paddle.device.device_guard(place_obj1):
                        xxx = paddle.randn([2, 2])  # create on place1
                        self.assertEqual(xxx.place, place_obj1)
                        self.assertNotEqual(xxx.place, place_obj2)

                        with paddle.device.device_guard(place_obj2):
                            xxxx = paddle.randn([2, 2])  # create on place1
                            self.assertEqual(xxxx.place, place_obj2)
                            self.assertNotEqual(xxxx.place, place_obj1)

                        self.assertEqual(xxxx.place, place_obj2)
                        self.assertNotEqual(xxxx.place, place_obj1)

                    self.assertEqual(xxx.place, place_obj1)
                    self.assertNotEqual(xxx.place, place_obj2)

                self.assertEqual(xx.place, place_obj2)
                self.assertNotEqual(xx.place, place_obj1)

            self.assertEqual(x.place, place_obj1)
            self.assertNotEqual(x.place, place_obj2)


if __name__ == "__main__":
    unittest.main()
