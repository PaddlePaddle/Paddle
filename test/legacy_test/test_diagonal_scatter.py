# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import convert_float_to_uint16

import paddle
from paddle import base
from paddle.base import core


def fill_diagonal_ndarray(x, value, offset=0, dim1=0, dim2=1):
    """Fill value into the diagonal of x that offset is ${offset} and the coordinate system is (dim1, dim2)."""
    strides = x.strides
    shape = x.shape
    if dim1 > dim2:
        dim1, dim2 = dim2, dim1
    assert 0 <= dim1 < dim2 <= 2
    assert len(x.shape) == 3

    dim_sum = dim1 + dim2
    dim3 = len(x.shape) - dim_sum
    if offset >= 0:
        diagdim = min(shape[dim1], shape[dim2] - offset)
        diagonal = np.lib.stride_tricks.as_strided(
            x[:, offset:] if dim_sum == 1 else x[:, :, offset:],
            shape=(shape[dim3], diagdim),
            strides=(strides[dim3], strides[dim1] + strides[dim2]),
        )
    else:
        diagdim = min(shape[dim2], shape[dim1] + offset)
        diagonal = np.lib.stride_tricks.as_strided(
            x[-offset:, :] if dim_sum in [1, 2] else x[:, -offset:],
            shape=(shape[dim3], diagdim),
            strides=(strides[dim3], strides[dim1] + strides[dim2]),
        )

    diagonal[...] = value
    return x


def fill_gt(x, y, offset, dim1, dim2):
    if dim1 > dim2:
        dim1, dim2 = dim2, dim1
        offset = -offset
    xshape = x.shape
    yshape = y.shape

    perm_list = []
    unperm_list = [0] * len(xshape)
    idx = 0

    for i in range(len(xshape)):
        if i != dim1 and i != dim2:
            perm_list.append(i)
            unperm_list[i] = idx
            idx += 1
    perm_list += [dim1, dim2]
    unperm_list[dim1] = idx
    unperm_list[dim2] = idx + 1

    x = np.transpose(x, perm_list)
    y = y.reshape((-1, yshape[-1]))
    nxshape = x.shape
    x = x.reshape((-1, xshape[dim1], xshape[dim2]))
    out = fill_diagonal_ndarray(x, y, offset, 1, 2)

    out = out.reshape(nxshape)
    out = np.transpose(out, unperm_list)
    return out


class TestDiagonalScatterAPI(unittest.TestCase):
    def set_args(self):
        self.dtype = "float32"
        self.x = np.random.random([10, 10]).astype(np.float32)
        self.y = np.random.random([10]).astype(np.float32)
        self.offset = 0
        self.axis1 = 0
        self.axis2 = 1

    def set_api(self):
        self.ref_api = fill_gt
        self.paddle_api = paddle.diagonal_scatter

    def get_output(self):
        self.output = self.ref_api(
            self.x, self.y, self.offset, self.axis1, self.axis2
        )

    def setUp(self):
        # init the test case
        self.set_api()
        self.set_args()
        self.get_output()

    def test_dygraph(self):
        paddle.disable_static()
        x = paddle.to_tensor(self.x, self.dtype)
        y = paddle.to_tensor(self.y, self.dtype)
        result = paddle.diagonal_scatter(
            x, y, offset=self.offset, axis1=self.axis1, axis2=self.axis2
        )
        np.testing.assert_allclose(self.output, result.numpy(), rtol=1e-5)
        paddle.enable_static()

    def test_static(self):
        if self.dtype not in [
            "float16",
            "float32",
            "float64",
            "int16",
            "int32",
            "int64",
            "bool",
            "uint16",
        ]:
            return
        paddle.enable_static()
        startup_program = base.Program()
        train_program = base.Program()
        with base.program_guard(startup_program, train_program):
            x = paddle.static.data(
                name="X", shape=self.x.shape, dtype=self.dtype
            )
            y = paddle.static.data(
                name="Y", shape=self.y.shape, dtype=self.dtype
            )
            out = paddle.diagonal_scatter(
                x, y, offset=self.offset, axis1=self.axis1, axis2=self.axis2
            )

            place = (
                base.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else base.CPUPlace()
            )

            exe = base.Executor(place)
            result = exe.run(
                base.default_main_program(),
                feed={"X": self.x, "Y": self.y},
                fetch_list=[out],
            )
            np.testing.assert_allclose(self.output, result[0], rtol=1e-5)
            paddle.disable_static()


# check the data type of the input
class TestDiagonalScatterFloat16(TestDiagonalScatterAPI):
    def set_args(self):
        self.dtype = "float16"
        self.x = np.random.random([10, 10]).astype(np.float16)
        self.y = np.random.random([10]).astype(np.float16)
        self.offset = 0
        self.axis1 = 0
        self.axis2 = 1


class TestDiagonalScatterFloat64(TestDiagonalScatterAPI):
    def set_args(self):
        self.dtype = "float64"
        self.x = np.random.random([10, 10]).astype(np.float64)
        self.y = np.random.random([10]).astype(np.float64)
        self.offset = 0
        self.axis1 = 0
        self.axis2 = 1


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support bfloat16",
)
class TestDiagonalScatterBFloat16(TestDiagonalScatterAPI):
    def set_args(self):
        self.dtype = "uint16"
        self.x = convert_float_to_uint16(
            np.random.random([10, 10]).astype(np.float32)
        )
        self.y = convert_float_to_uint16(
            np.random.random([10]).astype(np.float32)
        )
        self.offset = 0
        self.axis1 = 0
        self.axis2 = 1


class TestDiagoalScatterUInt8(TestDiagonalScatterAPI):
    def set_args(self):
        self.dtype = "uint8"
        self.x = np.random.randint(0, 255, [10, 10]).astype(np.uint8)
        self.y = np.random.randint(0, 255, [10]).astype(np.uint8)
        self.offset = 0
        self.axis1 = 0
        self.axis2 = 1


class TestDiagoalScatterInt8(TestDiagonalScatterAPI):
    def set_args(self):
        self.dtype = "int8"
        self.x = np.random.randint(-128, 127, [10, 10]).astype(np.int8)
        self.y = np.random.randint(-128, 127, [10]).astype(np.int8)
        self.offset = 0
        self.axis1 = 0
        self.axis2 = 1


class TestDiagoalScatterInt16(TestDiagonalScatterAPI):
    def set_args(self):
        self.dtype = "int16"
        self.x = np.random.randint(-128, 127, [10, 10]).astype(np.int16)
        self.y = np.random.randint(-128, 127, [10]).astype(np.int16)
        self.offset = 0
        self.axis1 = 0
        self.axis2 = 1


class TestDiagoalScatterInt32(TestDiagonalScatterAPI):
    def set_args(self):
        self.dtype = "int32"
        self.x = np.random.randint(-256, 255, [10, 10]).astype(np.int32)
        self.y = np.random.randint(-256, 255, [10]).astype(np.int32)
        self.offset = 0
        self.axis1 = 0
        self.axis2 = 1


class TestDiagoalScatterInt64(TestDiagonalScatterAPI):
    def set_args(self):
        self.dtype = "int64"
        self.x = np.random.randint(-1024, 1023, [10, 10]).astype(np.int64)
        self.y = np.random.randint(-1024, 1023, [10]).astype(np.int64)
        self.offset = 0
        self.axis1 = 0
        self.axis2 = 1


class TestDiagoalScatterBool(TestDiagonalScatterAPI):
    def set_args(self):
        self.dtype = "bool"
        self.x = np.random.randint(0, 1, [10, 10]).astype(np.bool_)
        self.y = np.random.randint(0, 1, [10]).astype(np.bool_)
        self.offset = 0
        self.axis1 = 0
        self.axis2 = 1


class TestDiagoalScatterComplex64(TestDiagonalScatterAPI):
    def set_args(self):
        self.dtype = "complex64"
        self.x = np.random.random([10, 10]).astype(np.float32)
        self.x = self.x + 1j * self.x
        self.y = np.random.random([10]).astype(np.float32)
        self.y = self.y + 1j * self.y
        self.offset = 0
        self.axis1 = 0
        self.axis2 = 1


class TestDiagoalScatterComplex128(TestDiagonalScatterAPI):
    def set_args(self):
        self.dtype = "complex128"
        self.x = np.random.random([10, 10]).astype(np.float64)
        self.x = self.x + 1j * self.x
        self.y = np.random.random([10]).astype(np.float64)
        self.y = self.y + 1j * self.y
        self.offset = 0
        self.axis1 = 0
        self.axis2 = 1


# check offset, axis
class TestDiagoalScatterOffset(TestDiagonalScatterAPI):
    def set_args(self):
        self.dtype = "float32"
        self.x = np.random.random([10, 10]).astype(np.float32)
        self.y = np.random.random([9]).astype(np.float32)
        self.offset = 1
        self.axis1 = 0
        self.axis2 = 1


class TestDiagoalScatterOffset2(TestDiagonalScatterAPI):
    def set_args(self):
        self.dtype = "float32"
        self.x = np.random.random([10, 10]).astype(np.float32)
        self.y = np.random.random([8]).astype(np.float32)
        self.offset = -2
        self.axis1 = 0
        self.axis2 = 1


class TestDiagoalScatterAxis1(TestDiagonalScatterAPI):
    def set_args(self):
        self.dtype = "float32"
        self.x = np.random.random([10, 10]).astype(np.float32)
        self.y = np.random.random([10]).astype(np.float32)
        self.offset = 0
        self.axis1 = 1
        self.axis2 = 0


# check error
class TestDiagonalScatterError(TestDiagonalScatterAPI):
    def test_tensor_x_dimension_error(self):
        # Tensor x must be at least 2-dimensional in diagonal_scatter
        paddle.disable_static()
        x = paddle.to_tensor([1.0], "float32")
        y = paddle.to_tensor([], "float32")
        with self.assertRaises(AssertionError):
            paddle.diagonal_scatter(x, y)
        paddle.enable_static()

    def test_tensor_y_dimension_error(self):
        # y.shape should be (10,), but received (1,)
        paddle.disable_static()
        x = paddle.to_tensor(self.x, self.dtype)
        y = paddle.to_tensor([1.0], "float32")
        with self.assertRaises(AssertionError):
            paddle.diagonal_scatter(x, y)
        paddle.enable_static()

    def test_axis1_out_of_range_error(self):
        # axis1 is out of range in diagonal_scatter (expected to be in range of [-2, 2), but got 1000)
        paddle.disable_static()
        x = paddle.to_tensor(self.x, self.dtype)
        y = paddle.to_tensor(self.y, self.dtype)
        axis1 = 1000
        with self.assertRaises(AssertionError):
            paddle.diagonal_scatter(x, y, self.offset, axis1, self.axis2)
        paddle.enable_static()

    def test_axis2_out_of_range_error(self):
        # axis2 is out of range in diagonal_scatter (expected to be in range of [-2, 2), but got -1000)
        paddle.disable_static()
        x = paddle.to_tensor(self.x, self.dtype)
        y = paddle.to_tensor(self.y, self.dtype)
        axis2 = -1000
        with self.assertRaises(AssertionError):
            paddle.diagonal_scatter(x, y, self.offset, self.axis1, axis2)
        paddle.enable_static()

    def test_axis1_axis2_be_identical_error(self):
        # axis1 and axis2 should not be identical in diagonal_scatter, but received axis1 = 0, axis2 = 0
        paddle.disable_static()
        x = paddle.to_tensor(self.x, self.dtype)
        y = paddle.to_tensor(self.y, self.dtype)
        axis1 = 0
        axis2 = 0
        with self.assertRaises(AssertionError):
            paddle.diagonal_scatter(x, y, self.offset, axis1, axis2)
        paddle.enable_static()


if __name__ == "__main__":
    unittest.main()
