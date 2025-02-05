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
    IrMode,
    ToStaticMode,
    disable_test_case,
    test_ast_only,
    test_pir_only,
    test_sot_only,
)

import paddle
from paddle import base

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
    "bool",
] + ([] if base.core.is_compiled_with_xpu() else ["complex64", "complex128"])

_cpu_place = "Place(cpu)"
_gpu_place = "Place(gpu:0)"
_xpu_place = "Place(xpu:0)"


def place_res():
    def res():
        if paddle.is_compiled_with_cuda():
            return _gpu_place
        elif paddle.is_compiled_with_xpu():
            return _xpu_place
        else:
            return _cpu_place

    return res


get_place = place_res()


def to_dtype(tensor_x, dtype):
    return tensor_x.to(dtype)


def to_device(tensor_x, device):
    return tensor_x.to(device)


def to__device(tensor_x, device):
    return tensor_x._to(device)


def to_device_dtype(tensor_x, device, dtype):
    return tensor_x.to(device, dtype)


def to_other(tensor_x, other):
    return tensor_x.to(other)


def to_other_blocking(tensor_x, other, blocking):
    return tensor_x.to(other, blocking)


def to_dtype_blocking(tensor_x, dtype, blocking):
    return tensor_x.to(dtype, blocking)


def to_device_dtype_blocking(tensor_x, device, dtype, blocking):
    return tensor_x.to(device, dtype, blocking)


def to_kwargs_tesnor_device(tensor_x, tensor_y):
    return tensor_x.to(device=tensor_y.place)


def to_kwargs_device_dtype_blocking(tensor_x, device, dtype, blocking):
    return tensor_x.to(device=device, dtype=dtype, blocking=blocking)


def to_kwargs_other(tensor_x, other):
    return tensor_x.to(other=other)


def to_invalid_key_error(tensor_x, device, dtype, test_key):
    return tensor_x.to(device, dtype, test_key=test_key)


def to_many_key_error(tensor_x, device, dtype):
    return tensor_x.to(device, dtype, device, dtype)


class TensorToTest(Dy2StTestBase):
    @test_pir_only
    def test_tensor_to_dtype(self):
        tensor_x = paddle.to_tensor([1, 2, 3])
        for dtype in _valid_dtypes:
            t = paddle.jit.to_static(to_dtype)(tensor_x, dtype)
            type_x_str = str(t.dtype)
            self.assertEqual(type_x_str, "paddle." + dtype)

    @test_pir_only
    def test_tensor_to_device(self):
        if paddle.is_compiled_with_cuda():
            x = paddle.to_tensor([1, 2, 3], place="gpu")
        elif paddle.is_compiled_with_xpu():
            x = paddle.to_tensor([1, 2, 3], place="xpu")
        else:
            x = paddle.to_tensor([1, 2, 3])

        y = paddle.to_tensor([1, 2, 3], place="cpu")
        y = paddle.jit.to_static(to_kwargs_tesnor_device)(y, x)
        self.assertEqual(str(x.place), str(y.place))

    @test_pir_only
    def test_tensor_to_device2(self):
        if paddle.is_compiled_with_cuda():
            x = paddle.to_tensor([1, 2, 3], place="gpu")
        elif paddle.is_compiled_with_xpu():
            x = paddle.to_tensor([1, 2, 3], place="xpu")
        else:
            x = paddle.to_tensor([1, 2, 3])

        y = paddle.to_tensor([1, 2, 3], place="cpu")

        y = paddle.jit.to_static(to_device)(y, x.place)
        self.assertEqual(str(x.place), str(y.place))

    @test_pir_only
    def test_tensor_to_device_dtype(self):
        tensor_x = paddle.to_tensor([1, 2, 3])
        places = ["cpu"]
        if paddle.is_compiled_with_cuda():
            places.append("gpu")
        if paddle.is_compiled_with_xpu():
            places.append("xpu")
        for dtype in _valid_dtypes:
            for place in places:
                tensor_x = paddle.jit.to_static(to_device_dtype)(
                    tensor_x, place, dtype
                )
                place_x_str = str(tensor_x.place)
                if "gpu" == place:
                    self.assertEqual(place_x_str, _gpu_place)
                elif "xpu" == place:
                    self.assertEqual(place_x_str, _xpu_place)
                else:
                    self.assertEqual(place_x_str, _cpu_place)
                type_x_str = str(tensor_x.dtype)
                self.assertEqual(type_x_str, "paddle." + dtype)

    # TODO(gouzil): Fix MIN_GRAPH_SIZE=10 case
    @test_pir_only
    @disable_test_case((ToStaticMode.SOT_MGS10, IrMode.PIR))
    def test_tensor_to_blocking(self):
        tensor_x = paddle.to_tensor([1, 2, 3])
        tensor_x = paddle.jit.to_static(to_device_dtype_blocking)(
            tensor_x, "cpu", "int32", False
        )
        self.assertEqual(str(tensor_x.place), _cpu_place)
        self.assertEqual(tensor_x.dtype, paddle.int32)
        tensor2 = paddle.to_tensor([4, 5, 6])
        tensor2 = paddle.jit.to_static(to_other_blocking)(
            tensor2, tensor_x, False
        )
        # Note: in static mode, the place of tensor2 is not changed
        self.assertEqual(str(tensor2.place), get_place())
        self.assertEqual(tensor2.dtype, paddle.int32)
        tensor2 = paddle.jit.to_static(to_dtype_blocking)(
            tensor2, "float16", False
        )
        self.assertEqual(tensor2.dtype, paddle.float16)

    @test_pir_only
    @disable_test_case((ToStaticMode.SOT_MGS10, IrMode.PIR))
    def test_tensor_to_other(self):
        tensor1 = paddle.to_tensor([1, 2, 3], dtype="int8", place="cpu")
        tensor2 = paddle.to_tensor([1, 2, 3])
        tensor2 = paddle.jit.to_static(to_other)(tensor2, tensor1)
        self.assertEqual(tensor2.dtype, tensor1.dtype)
        # Note: in static mode, the place of tensor2 is not changed
        self.assertEqual(str(tensor1.place), _cpu_place)
        self.assertEqual(str(tensor2.place), get_place())

    @test_pir_only
    @disable_test_case((ToStaticMode.SOT_MGS10, IrMode.PIR))
    def test_kwargs(self):
        tensor_x = paddle.to_tensor([1, 2, 3])
        tensor_x = paddle.jit.to_static(to_kwargs_device_dtype_blocking)(
            tensor_x, device="cpu", dtype="int8", blocking=True
        )
        self.assertEqual(str(tensor_x.place), _cpu_place)
        self.assertEqual(tensor_x.dtype, paddle.int8)
        tensor2 = paddle.to_tensor([4, 5, 6])
        tensor2 = paddle.jit.to_static(to_kwargs_other)(tensor2, other=tensor_x)
        # Note: in static mode, the place of tensor2 is not changed
        self.assertEqual(str(tensor2.place), get_place())
        self.assertEqual(tensor2.dtype, paddle.int8)

    @test_ast_only
    @test_pir_only
    def test_ast_error(self):
        tensor_x = paddle.to_tensor([1, 2, 3])
        # device value error
        with self.assertRaises(ValueError) as context1:
            paddle.jit.to_static(to_device)(tensor_x, "error_device")
        self.assertTrue(
            "The device must be a string which is like"
            in str(context1.exception)
        )
        # no matching signature error
        with self.assertRaises(ValueError) as context2:
            paddle.jit.to_static(to_device)(tensor_x, int)
        self.assertTrue(
            "No matching signature found" in str(context2.exception)
        )
        # invalid key error
        with self.assertRaises(TypeError) as context3:
            paddle.jit.to_static(to_invalid_key_error)(
                tensor_x, "cpu", "int32", test_key=False
            )
        self.assertTrue(
            "to() got an unexpected keyword argument" in str(context3.exception)
        )
        # device value error
        with self.assertRaises(ValueError) as context4:
            paddle.jit.to_static(to__device)(tensor_x, int)
        self.assertTrue(
            "device value error, must be str" in str(context4.exception)
        )
        # too many key error
        with self.assertRaises(TypeError) as context5:
            paddle.jit.to_static(to_many_key_error)(tensor_x, "cpu", "int32")
        self.assertTrue(
            "to() received too many arguments" in str(context5.exception)
        )

    @test_sot_only
    @test_pir_only
    def test_sot_error(self):
        tensor_x = paddle.to_tensor([1, 2, 3])
        # device value error
        with self.assertRaises(Exception) as context1:
            paddle.jit.to_static(to_device)(tensor_x, "error_device")
        self.assertTrue(
            "The device must be a string which is like"
            in str(context1.exception)
        )
        # no matching signature error
        with self.assertRaises(Exception) as context2:
            paddle.jit.to_static(to_device)(tensor_x, int)
        self.assertTrue(
            "No matching signature found" in str(context2.exception)
        )
        # invalid key error
        with self.assertRaises(Exception) as context3:
            paddle.jit.to_static(to_invalid_key_error)(
                tensor_x, "cpu", "int32", test_key=False
            )
        self.assertTrue(
            "to() got an unexpected keyword argument" in str(context3.exception)
        )
        # device value error
        with self.assertRaises(Exception) as context4:
            paddle.jit.to_static(to__device)(tensor_x, int)
        self.assertTrue(
            "device value error, must be str" in str(context4.exception)
        )
        # too many key error
        with self.assertRaises(Exception) as context5:
            paddle.jit.to_static(to_many_key_error)(tensor_x, "cpu", "int32")
        self.assertTrue(
            "to() received too many arguments" in str(context5.exception)
        )


if __name__ == '__main__':
    unittest.main()
