# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import base


class TensorToTest(unittest.TestCase):
    def test_Tensor_to_dtype(self):
        tensorx = paddle.to_tensor([1, 2, 3])
        valid_dtypes = [
            "bfloat16",
            "float16",
            "float32",
            "float64",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "bool",
        ] + (
            []
            if base.core.is_compiled_with_xpu()
            else ["complex64", "complex128"]
        )
        for dtype in valid_dtypes:
            tensorx = tensorx.to(dtype)
            typex_str = str(tensorx.dtype)
            self.assertTrue(typex_str, "paddle." + dtype)

    def test_Tensor_to_device(self):
        tensorx = paddle.to_tensor([1, 2, 3])
        places = ["cpu"]
        if base.core.is_compiled_with_cuda():
            places.append("gpu:0")
            places.append("gpu")
        if base.core.is_compiled_with_xpu():
            places.append("xpu:0")
            places.append("xpu")

        for place in places:
            tensorx = tensorx.to(place)
            placex_str = str(tensorx.place)
            if place == "gpu" or place == "xpu":
                self.assertTrue(placex_str, "Place(" + place + ":0)")
            else:
                self.assertTrue(placex_str, "Place(" + place + ")")

    def test_Tensor_to_device2(self):
        x = paddle.to_tensor([1, 2, 3])
        y = paddle.to_tensor([1, 2, 3], place="cpu")

        y.to(x.place)
        self.assertTrue(x.place, y.place)

    def test_Tensor_to_device_dtype(self):
        tensorx = paddle.to_tensor([1, 2, 3])
        places = ["cpu"]
        if base.core.is_compiled_with_cuda():
            places.append("gpu:0")
            places.append("gpu")
        if base.core.is_compiled_with_xpu():
            places.append("xpu:0")
            places.append("xpu")
        valid_dtypes = [
            "bfloat16",
            "float16",
            "float32",
            "float64",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "bool",
        ] + (
            []
            if base.core.is_compiled_with_xpu()
            else ["complex64", "complex128"]
        )
        for dtype in valid_dtypes:
            for place in places:
                tensorx = tensorx.to(place, dtype)
                placex_str = str(tensorx.place)
                if place == "gpu" or place == "xpu":
                    self.assertTrue(placex_str, "Place(" + place + ":0)")
                else:
                    self.assertTrue(placex_str, "Place(" + place + ")")
                typex_str = str(tensorx.dtype)
                self.assertTrue(typex_str, "paddle." + dtype)

    def test_Tensor_to_blocking(self):
        tensorx = paddle.to_tensor([1, 2, 3])
        tensorx = tensorx.to("cpu", "int32", False)
        placex_str = str(tensorx.place)
        self.assertTrue(placex_str, "Place(cpu)")
        typex_str = str(tensorx.dtype)
        self.assertTrue(typex_str, "paddle.int32")
        tensor2 = paddle.to_tensor([4, 5, 6])
        tensor2 = tensor2.to(tensorx, False)
        place2_str = str(tensor2.place)
        self.assertTrue(place2_str, "Place(cpu)")
        type2_str = str(tensor2.dtype)
        self.assertTrue(type2_str, "paddle.int32")
        tensor2 = tensor2.to("float16", False)
        type2_str = str(tensor2.dtype)
        self.assertTrue(type2_str, "paddle.float16")

    def test_Tensor_to_other(self):
        tensor1 = paddle.to_tensor([1, 2, 3], dtype="int8", place="cpu")
        tensor2 = paddle.to_tensor([1, 2, 3])
        tensor2 = tensor2.to(tensor1)
        self.assertTrue(tensor2.dtype, tensor1.dtype)
        self.assertTrue(type(tensor2.place), type(tensor1.place))

    def test_kwargs(self):
        tensorx = paddle.to_tensor([1, 2, 3])
        tensorx = tensorx.to(device="cpu", dtype="int8", blocking=True)
        placex_str = str(tensorx.place)
        self.assertTrue(placex_str, "Place(cpu)")
        typex_str = str(tensorx.dtype)
        self.assertTrue(typex_str, "paddle.int8")
        tensor2 = paddle.to_tensor([4, 5, 6])
        tensor2 = tensor2.to(other=tensorx)
        place2_str = str(tensor2.place)
        self.assertTrue(place2_str, "Place(cpu)")
        type2_str = str(tensor2.dtype)
        self.assertTrue(type2_str, "paddle.int8")

    def test_error(self):
        tensorx = paddle.to_tensor([1, 2, 3])
        # device value error
        try:
            tensorx = tensorx.to("error_device")
        except Exception as error:
            self.assertIsInstance(error, ValueError)
        # to many augments
        try:
            tensorx = tensorx.to("cpu", "int32", False, "test_aug")
        except Exception as error:
            self.assertIsInstance(error, TypeError)
        # invalid key
        try:
            tensorx = tensorx.to("cpu", "int32", test_key=False)
        except Exception as error:
            self.assertIsInstance(error, TypeError)


if __name__ == '__main__':
    unittest.main()
