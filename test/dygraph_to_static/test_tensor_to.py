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

from dygraph_to_static_utils import (
    Dy2StTestBase,
    test_ast_only,
    test_pir_only,
    test_sot_only,
)

import paddle

# NOTE: only test in PIR mode

_valid_dtypes = [
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "complex64",
    "complex128",
    "bool",
]

_cpu_place = "Place(cpu)"
_gpu_place = "Place(gpu:0)"


def place_res():
    def res():
        if paddle.is_compiled_with_cuda():
            return _gpu_place
        else:
            return _cpu_place

    return res


get_place = place_res()


def to_dtype(tensorx, dtype):
    return tensorx.to(dtype)


def to_device(tensorx, device):
    return tensorx.to(device)


def to__device(tensorx, device):
    return tensorx._to(device)


def to_device_dtype(tensorx, device, dtype):
    return tensorx.to(device, dtype)


def to_other(tensorx, other):
    return tensorx.to(other)


def to_other_blocking(tensorx, other, blocking):
    return tensorx.to(other, blocking)


def to_dtype_blocking(tensorx, dtype, blocking):
    return tensorx.to(dtype, blocking)


def to_device_dtype_blocking(tensorx, device, dtype, blocking):
    return tensorx.to(device, dtype, blocking)


def to_kwargs_device_dtype_blocking(tensorx, device, dtype, blocking):
    return tensorx.to(device=device, dtype=dtype, blocking=blocking)


def to_kwargs_other(tensorx, other):
    return tensorx.to(other=other)


def to_invalid_key_error(tensorx, device, dtype, test_key):
    return tensorx.to(device, dtype, test_key=test_key)


def to_many_key_error(tensorx, device, dtype):
    return tensorx.to(device, dtype, device, dtype)


class TensorToTest(Dy2StTestBase):
    @test_pir_only
    def test_tensor_to_dtype(self):
        tensorx = paddle.to_tensor([1, 2, 3])
        for dtype in _valid_dtypes:
            t = paddle.jit.to_static(to_dtype)(tensorx, dtype)
            typex_str = str(t.dtype)
            self.assertEqual(typex_str, "paddle." + dtype)

    @test_pir_only
    def test_tensor_to_device2(self):
        if paddle.is_compiled_with_cuda():
            x = paddle.to_tensor([1, 2, 3], place="gpu")
        else:
            x = paddle.to_tensor([1, 2, 3])

        y = paddle.to_tensor([1, 2, 3], place="cpu")

        y = paddle.jit.to_static(to_device)(y, x.place)
        self.assertEqual(str(x.place), str(y.place))

    @test_pir_only
    def test_tensor_to_device_dtype(self):
        tensorx = paddle.to_tensor([1, 2, 3])
        places = ["cpu"]
        if paddle.is_compiled_with_cuda():
            places.append("gpu")
        for dtype in _valid_dtypes:
            for place in places:
                tensorx = paddle.jit.to_static(to_device_dtype)(
                    tensorx, place, dtype
                )
                placex_str = str(tensorx.place)
                if "gpu" == place:
                    self.assertEqual(placex_str, _gpu_place)
                else:
                    self.assertEqual(placex_str, _cpu_place)
                typex_str = str(tensorx.dtype)
                self.assertEqual(typex_str, "paddle." + dtype)

    @test_pir_only
    def test_tensor_to_blocking(self):
        tensorx = paddle.to_tensor([1, 2, 3])
        tensorx = paddle.jit.to_static(to_device_dtype_blocking)(
            tensorx, "cpu", "int32", False
        )
        self.assertEqual(str(tensorx.place), _cpu_place)
        self.assertEqual(tensorx.dtype, paddle.int32)
        tensor2 = paddle.to_tensor([4, 5, 6])
        tensor2 = paddle.jit.to_static(to_other_blocking)(
            tensor2, tensorx, False
        )
        # Note: in static mode, the place of tensor2 is not changed
        self.assertEqual(str(tensor2.place), get_place())
        self.assertEqual(tensor2.dtype, paddle.int32)
        tensor2 = paddle.jit.to_static(to_dtype_blocking)(
            tensor2, "float16", False
        )
        self.assertEqual(tensor2.dtype, paddle.float16)

    @test_pir_only
    def test_tensor_to_other(self):
        tensor1 = paddle.to_tensor([1, 2, 3], dtype="int8", place="cpu")
        tensor2 = paddle.to_tensor([1, 2, 3])
        tensor2 = paddle.jit.to_static(to_other)(tensor2, tensor1)
        self.assertEqual(tensor2.dtype, tensor1.dtype)
        # Note: in static mode, the place of tensor2 is not changed
        self.assertEqual(str(tensor1.place), _cpu_place)
        self.assertEqual(str(tensor2.place), get_place())

    @test_pir_only
    def test_kwargs(self):
        tensorx = paddle.to_tensor([1, 2, 3])
        tensorx = paddle.jit.to_static(to_kwargs_device_dtype_blocking)(
            tensorx, device="cpu", dtype="int8", blocking=True
        )
        self.assertEqual(str(tensorx.place), _cpu_place)
        self.assertEqual(tensorx.dtype, paddle.int8)
        tensor2 = paddle.to_tensor([4, 5, 6])
        tensor2 = paddle.jit.to_static(to_kwargs_other)(tensor2, other=tensorx)
        # Note: in static mode, the place of tensor2 is not changed
        self.assertEqual(str(tensor2.place), get_place())
        self.assertEqual(tensor2.dtype, paddle.int8)

    @test_ast_only
    @test_pir_only
    def test_ast_error(self):
        tensorx = paddle.to_tensor([1, 2, 3])
        # device value error
        with self.assertRaises(ValueError) as context1:
            paddle.jit.to_static(to_device)(tensorx, "error_device")
        self.assertTrue(
            "The device must be a string which is like"
            in str(context1.exception)
        )
        # no matching signature error
        with self.assertRaises(ValueError) as context2:
            paddle.jit.to_static(to_device)(tensorx, int)
        self.assertTrue(
            "No matching signature found" in str(context2.exception)
        )
        # invalid key error
        with self.assertRaises(TypeError) as context3:
            paddle.jit.to_static(to_invalid_key_error)(
                tensorx, "cpu", "int32", test_key=False
            )
        self.assertTrue(
            "to() got an unexpected keyword argument" in str(context3.exception)
        )
        # device value error
        with self.assertRaises(ValueError) as context4:
            paddle.jit.to_static(to__device)(tensorx, int)
        self.assertTrue(
            "device value error, must be str" in str(context4.exception)
        )
        # too many key error
        with self.assertRaises(TypeError) as context5:
            paddle.jit.to_static(to_many_key_error)(tensorx, "cpu", "int32")
        self.assertTrue(
            "to() received too many arguments" in str(context5.exception)
        )

    @test_sot_only
    @test_pir_only
    def test_sot_error(self):
        tensorx = paddle.to_tensor([1, 2, 3])
        # device value error
        with self.assertRaises(Exception) as context1:
            paddle.jit.to_static(to_device)(tensorx, "error_device")
        self.assertTrue(
            "The device must be a string which is like"
            in str(context1.exception)
        )
        # no matching signature error
        with self.assertRaises(Exception) as context2:
            paddle.jit.to_static(to_device)(tensorx, int)
        self.assertTrue(
            "No matching signature found" in str(context2.exception)
        )
        # invalid key error
        with self.assertRaises(Exception) as context3:
            paddle.jit.to_static(to_invalid_key_error)(
                tensorx, "cpu", "int32", test_key=False
            )
        self.assertTrue(
            "to() got an unexpected keyword argument" in str(context3.exception)
        )
        # device value error
        with self.assertRaises(Exception) as context4:
            paddle.jit.to_static(to__device)(tensorx, int)
        self.assertTrue(
            "device value error, must be str" in str(context4.exception)
        )
        # too many key error
        with self.assertRaises(Exception) as context5:
            paddle.jit.to_static(to_many_key_error)(tensorx, "cpu", "int32")
        self.assertTrue(
            "to() received too many arguments" in str(context5.exception)
        )


if __name__ == '__main__':
    unittest.main()
