#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest

import paddle
from paddle.base import core

paddle.enable_static()


# ----------------- TEST OP: BitwiseAnd ----------------- #
class TestBitwiseAnd(OpTest):
    def setUp(self):
        self.op_type = "bitwise_and"
        self.python_api = paddle.tensor.logic.bitwise_and
        self.init_dtype()
        self.init_shape()
        self.init_bound()

        x = np.random.randint(
            self.low, self.high, self.x_shape, dtype=self.dtype
        )
        y = np.random.randint(
            self.low, self.high, self.y_shape, dtype=self.dtype
        )
        out = np.bitwise_and(x, y)

        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(
            check_cinn=True, check_pir=True, check_symbol_infer=False
        )

    def test_check_grad(self):
        pass

    def init_dtype(self):
        self.dtype = np.int32

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [2, 3, 4, 5]

    def init_bound(self):
        self.low = -100
        self.high = 100


class TestBitwiseAnd_ZeroDim1(TestBitwiseAnd):
    def init_shape(self):
        self.x_shape = []
        self.y_shape = []


class TestBitwiseAnd_ZeroDim2(TestBitwiseAnd):
    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = []


class TestBitwiseAnd_ZeroDim3(TestBitwiseAnd):
    def init_shape(self):
        self.x_shape = []
        self.y_shape = [2, 3, 4, 5]


class TestBitwiseAndUInt8(TestBitwiseAnd):
    def init_dtype(self):
        self.dtype = np.uint8

    def init_bound(self):
        self.low = 0
        self.high = 100


class TestBitwiseAndInt8(TestBitwiseAnd):
    def init_dtype(self):
        self.dtype = np.int8

    def init_shape(self):
        self.x_shape = [4, 5]
        self.y_shape = [2, 3, 4, 5]


class TestBitwiseAndInt16(TestBitwiseAnd):
    def init_dtype(self):
        self.dtype = np.int16

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [4, 1]


class TestBitwiseAndInt64(TestBitwiseAnd):
    def init_dtype(self):
        self.dtype = np.int64

    def init_shape(self):
        self.x_shape = [1, 4, 1]
        self.y_shape = [2, 3, 4, 5]


class TestBitwiseAndBool(TestBitwiseAnd):
    def setUp(self):
        self.op_type = "bitwise_and"
        self.python_api = paddle.tensor.logic.bitwise_and

        self.init_shape()

        x = np.random.choice([True, False], self.x_shape)
        y = np.random.choice([True, False], self.y_shape)
        out = np.bitwise_and(x, y)

        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out}


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestElementwiseBitwiseAndOp_Stride(OpTest):
    no_need_check_grad = True

    def setUp(self):
        self.op_type = "bitwise_and"
        self.python_api = paddle.tensor.logic.bitwise_and
        self.public_python_api = paddle.tensor.logic.bitwise_and
        self.transpose_api = paddle.transpose
        self.as_stride_api = paddle.as_strided
        self.init_dtype()
        self.init_bound()
        self.init_input_output()

        self.inputs_stride = {
            'X': self.x,
            'Y': self.y_trans,
        }

        self.inputs = {
            'X': self.x,
            'Y': self.y,
        }

        self.outputs = {'Out': self.out}

    def init_dtype(self):
        self.dtype = np.int32

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_strided_forward = True
        self.check_output_with_place(
            place,
        )

    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.randint(
            self.low, self.high, [13, 17], dtype=self.dtype
        )
        self.y = np.random.randint(
            self.low, self.high, [13, 17], dtype=self.dtype
        )
        self.out = np.bitwise_and(self.x, self.y)
        self.perm = [1, 0]
        self.y_trans = np.transpose(self.y, self.perm)

    def init_bound(self):
        self.low = -100
        self.high = 100

    def test_check_grad(self):
        pass


class TestElementwiseBitwiseAndOp_Stride1(TestElementwiseBitwiseAndOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.randint(
            self.low, self.high, [20, 2, 13, 17], dtype=self.dtype
        )
        self.y = np.random.randint(
            self.low, self.high, [20, 2, 13, 17], dtype=self.dtype
        )
        self.out = np.bitwise_and(self.x, self.y)
        self.perm = [0, 1, 3, 2]
        self.y_trans = np.transpose(self.y, self.perm)


class TestElementwiseBitwiseAndOp_Stride2(TestElementwiseBitwiseAndOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.randint(
            self.low, self.high, [20, 2, 13, 17], dtype=self.dtype
        )
        self.y = np.random.randint(
            self.low, self.high, [20, 2, 13, 17], dtype=self.dtype
        )
        self.out = np.bitwise_and(self.x, self.y)
        self.perm = [0, 2, 1, 3]
        self.y_trans = np.transpose(self.y, self.perm)


class TestElementwiseBitwiseAndOp_Stride3(TestElementwiseBitwiseAndOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.randint(
            self.low, self.high, [20, 2, 13, 17], dtype=self.dtype
        )
        self.y = np.random.randint(
            self.low, self.high, [20, 2, 13, 1], dtype=self.dtype
        )
        self.out = np.bitwise_and(self.x, self.y)
        self.perm = [0, 1, 3, 2]
        self.y_trans = np.transpose(self.y, self.perm)


class TestElementwiseBitwiseAndOp_Stride4(TestElementwiseBitwiseAndOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.randint(
            self.low, self.high, [1, 2, 13, 17], dtype=self.dtype
        )
        self.y = np.random.randint(
            self.low, self.high, [20, 2, 13, 1], dtype=self.dtype
        )
        self.out = np.bitwise_and(self.x, self.y)
        self.perm = [1, 0, 2, 3]
        self.y_trans = np.transpose(self.y, self.perm)


class TestElementwiseBitwiseAndOp_Stride5(TestElementwiseBitwiseAndOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "as_stride"
        self.x = np.random.randint(
            self.low, self.high, [23, 10, 1, 17], dtype=self.dtype
        )
        self.y = np.random.randint(
            self.low, self.high, [23, 2, 13, 20], dtype=self.dtype
        )
        self.y_trans = self.y
        self.y = self.y[:, 0:1, :, 0:1]
        self.out = np.bitwise_and(self.x, self.y)
        self.shape_param = [23, 1, 13, 1]
        self.stride_param = [520, 260, 20, 1]


class TestElementwiseBitwiseAndOp_Stride_ZeroDim1(
    TestElementwiseBitwiseAndOp_Stride
):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.randint(self.low, self.high, [], dtype=self.dtype)
        self.y = np.random.randint(
            self.low, self.high, [13, 17], dtype=self.dtype
        )
        self.out = np.bitwise_and(self.x, self.y)
        self.perm = [1, 0]
        self.y_trans = np.transpose(self.y, self.perm)


class TestElementwiseBitwiseAndOp_Stride_ZeroSize1(
    TestElementwiseBitwiseAndOp_Stride
):
    def init_data(self):
        self.strided_input_type = "transpose"
        self.x = np.random.rand(1, 0, 2).astype('float32')
        self.y = np.random.rand(3, 0, 1).astype('float32')
        self.out = np.bitwise_and(self.x, self.y)
        self.perm = [2, 1, 0]
        self.y_trans = np.transpose(self.y, self.perm)


# ----------------- TEST OP: BitwiseOr ------------------ #
class TestBitwiseOr(OpTest):
    def setUp(self):
        self.op_type = "bitwise_or"
        self.python_api = paddle.tensor.logic.bitwise_or
        self.init_dtype()
        self.init_shape()
        self.init_bound()

        x = np.random.randint(
            self.low, self.high, self.x_shape, dtype=self.dtype
        )
        y = np.random.randint(
            self.low, self.high, self.y_shape, dtype=self.dtype
        )
        out = np.bitwise_or(x, y)

        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(
            check_cinn=True, check_pir=True, check_symbol_infer=False
        )

    def test_check_grad(self):
        pass

    def init_dtype(self):
        self.dtype = np.int32

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [2, 3, 4, 5]

    def init_bound(self):
        self.low = -100
        self.high = 100


class TestBitwiseOr_ZeroDim1(TestBitwiseOr):
    def init_shape(self):
        self.x_shape = []
        self.y_shape = []


class TestBitwiseOr_ZeroDim2(TestBitwiseOr):
    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = []


class TestBitwiseOr_ZeroDim3(TestBitwiseOr):
    def init_shape(self):
        self.x_shape = []
        self.y_shape = [2, 3, 4, 5]


class TestBitwiseOrUInt8(TestBitwiseOr):
    def init_dtype(self):
        self.dtype = np.uint8

    def init_bound(self):
        self.low = 0
        self.high = 100


class TestBitwiseOrInt8(TestBitwiseOr):
    def init_dtype(self):
        self.dtype = np.int8

    def init_shape(self):
        self.x_shape = [4, 5]
        self.y_shape = [2, 3, 4, 5]


class TestBitwiseOrInt16(TestBitwiseOr):
    def init_dtype(self):
        self.dtype = np.int16

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [4, 1]


class TestBitwiseOrInt64(TestBitwiseOr):
    def init_dtype(self):
        self.dtype = np.int64

    def init_shape(self):
        self.x_shape = [1, 4, 1]
        self.y_shape = [2, 3, 4, 5]


class TestBitwiseOrBool(TestBitwiseOr):
    def setUp(self):
        self.op_type = "bitwise_or"
        self.python_api = paddle.tensor.logic.bitwise_or

        self.init_shape()

        x = np.random.choice([True, False], self.x_shape)
        y = np.random.choice([True, False], self.y_shape)
        out = np.bitwise_or(x, y)

        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out}


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestElementwiseBitwiseOrOp_Stride(OpTest):
    no_need_check_grad = True

    def setUp(self):
        self.op_type = "bitwise_or"
        self.python_api = paddle.tensor.logic.bitwise_or
        self.public_python_api = paddle.tensor.logic.bitwise_or
        self.transpose_api = paddle.transpose
        self.as_stride_api = paddle.as_strided
        self.init_dtype()
        self.init_bound()
        self.init_input_output()

        self.inputs_stride = {
            'X': self.x,
            'Y': self.y_trans,
        }

        self.inputs = {
            'X': self.x,
            'Y': self.y,
        }

        self.outputs = {'Out': self.out}

    def init_dtype(self):
        self.dtype = np.int32

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_strided_forward = True
        self.check_output_with_place(
            place,
        )

    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.randint(
            self.low, self.high, [13, 17], dtype=self.dtype
        )
        self.y = np.random.randint(
            self.low, self.high, [13, 17], dtype=self.dtype
        )
        self.out = np.bitwise_or(self.x, self.y)
        self.perm = [1, 0]
        self.y_trans = np.transpose(self.y, self.perm)

    def init_bound(self):
        self.low = -100
        self.high = 100

    def test_check_grad(self):
        pass


class TestElementwiseBitwiseOrOp_Stride1(TestElementwiseBitwiseOrOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.randint(
            self.low, self.high, [20, 2, 13, 17], dtype=self.dtype
        )
        self.y = np.random.randint(
            self.low, self.high, [20, 2, 13, 17], dtype=self.dtype
        )
        self.out = np.bitwise_or(self.x, self.y)
        self.perm = [0, 1, 3, 2]
        self.y_trans = np.transpose(self.y, self.perm)


class TestElementwiseBitwiseOrOp_Stride2(TestElementwiseBitwiseOrOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.randint(
            self.low, self.high, [20, 2, 13, 17], dtype=self.dtype
        )
        self.y = np.random.randint(
            self.low, self.high, [20, 2, 13, 17], dtype=self.dtype
        )
        self.out = np.bitwise_or(self.x, self.y)
        self.perm = [0, 2, 1, 3]
        self.y_trans = np.transpose(self.y, self.perm)


class TestElementwiseBitwiseOrOp_Stride3(TestElementwiseBitwiseOrOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.randint(
            self.low, self.high, [20, 2, 13, 17], dtype=self.dtype
        )
        self.y = np.random.randint(
            self.low, self.high, [20, 2, 13, 1], dtype=self.dtype
        )
        self.out = np.bitwise_or(self.x, self.y)
        self.perm = [0, 1, 3, 2]
        self.y_trans = np.transpose(self.y, self.perm)


class TestElementwiseBitwiseOrOp_Stride4(TestElementwiseBitwiseOrOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.randint(
            self.low, self.high, [1, 2, 13, 17], dtype=self.dtype
        )
        self.y = np.random.randint(
            self.low, self.high, [20, 2, 13, 1], dtype=self.dtype
        )
        self.out = np.bitwise_or(self.x, self.y)
        self.perm = [1, 0, 2, 3]
        self.y_trans = np.transpose(self.y, self.perm)


class TestElementwiseBitwiseOrOp_Stride5(TestElementwiseBitwiseOrOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "as_stride"
        self.x = np.random.randint(
            self.low, self.high, [23, 10, 1, 17], dtype=self.dtype
        )
        self.y = np.random.randint(
            self.low, self.high, [23, 2, 13, 20], dtype=self.dtype
        )
        self.y_trans = self.y
        self.y = self.y[:, 0:1, :, 0:1]
        self.out = np.bitwise_or(self.x, self.y)
        self.shape_param = [23, 1, 13, 1]
        self.stride_param = [520, 260, 20, 1]


class TestElementwiseBitwiseOrOp_Stride_ZeroDim1(
    TestElementwiseBitwiseOrOp_Stride
):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.randint(self.low, self.high, [], dtype=self.dtype)
        self.y = np.random.randint(
            self.low, self.high, [13, 17], dtype=self.dtype
        )
        self.out = np.bitwise_or(self.x, self.y)
        self.perm = [1, 0]
        self.y_trans = np.transpose(self.y, self.perm)


class TestElementwiseBitwiseOrOp_Stride_ZeroSize1(
    TestElementwiseBitwiseOrOp_Stride
):
    def init_data(self):
        self.strided_input_type = "transpose"
        self.x = np.random.rand(1, 0, 2).astype('float32')
        self.y = np.random.rand(3, 0, 1).astype('float32')
        self.out = np.bitwise_or(self.x, self.y)
        self.perm = [2, 1, 0]
        self.y_trans = np.transpose(self.y, self.perm)


# ----------------- TEST OP: BitwiseXor ---------------- #
class TestBitwiseXor(OpTest):
    def setUp(self):
        self.op_type = "bitwise_xor"
        self.python_api = paddle.tensor.logic.bitwise_xor

        self.init_dtype()
        self.init_shape()
        self.init_bound()

        x = np.random.randint(
            self.low, self.high, self.x_shape, dtype=self.dtype
        )
        y = np.random.randint(
            self.low, self.high, self.y_shape, dtype=self.dtype
        )
        out = np.bitwise_xor(x, y)

        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(
            check_cinn=True, check_pir=True, check_symbol_infer=False
        )

    def test_check_grad(self):
        pass

    def init_dtype(self):
        self.dtype = np.int32

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [2, 3, 4, 5]

    def init_bound(self):
        self.low = -100
        self.high = 100


class TestBitwiseXor_ZeroDim1(TestBitwiseXor):
    def init_shape(self):
        self.x_shape = []
        self.y_shape = []


class TestBitwiseXor_ZeroDim2(TestBitwiseXor):
    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = []


class TestBitwiseXor_ZeroDim3(TestBitwiseXor):
    def init_shape(self):
        self.x_shape = []
        self.y_shape = [2, 3, 4, 5]


class TestBitwiseXorUInt8(TestBitwiseXor):
    def init_dtype(self):
        self.dtype = np.uint8

    def init_bound(self):
        self.low = 0
        self.high = 100


class TestBitwiseXorInt8(TestBitwiseXor):
    def init_dtype(self):
        self.dtype = np.int8

    def init_shape(self):
        self.x_shape = [4, 5]
        self.y_shape = [2, 3, 4, 5]


class TestBitwiseXorInt16(TestBitwiseXor):
    def init_dtype(self):
        self.dtype = np.int16

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]
        self.y_shape = [4, 1]


class TestBitwiseXorInt64(TestBitwiseXor):
    def init_dtype(self):
        self.dtype = np.int64

    def init_shape(self):
        self.x_shape = [1, 4, 1]
        self.y_shape = [2, 3, 4, 5]


class TestBitwiseXorBool(TestBitwiseXor):
    def setUp(self):
        self.op_type = "bitwise_xor"
        self.python_api = paddle.tensor.logic.bitwise_xor

        self.init_shape()

        x = np.random.choice([True, False], self.x_shape)
        y = np.random.choice([True, False], self.y_shape)
        out = np.bitwise_xor(x, y)

        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out}


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestElementwiseBitwiseXorOp_Stride(OpTest):
    no_need_check_grad = True

    def setUp(self):
        self.op_type = "bitwise_xor"
        self.python_api = paddle.tensor.logic.bitwise_xor
        self.public_python_api = paddle.tensor.logic.bitwise_xor
        self.transpose_api = paddle.transpose
        self.as_stride_api = paddle.as_strided
        self.init_dtype()
        self.init_bound()
        self.init_input_output()

        self.inputs_stride = {
            'X': self.x,
            'Y': self.y_trans,
        }

        self.inputs = {
            'X': self.x,
            'Y': self.y,
        }

        self.outputs = {'Out': self.out}

    def init_dtype(self):
        self.dtype = np.int32

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_strided_forward = True
        self.check_output_with_place(
            place,
        )

    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.randint(
            self.low, self.high, [13, 17], dtype=self.dtype
        )
        self.y = np.random.randint(
            self.low, self.high, [13, 17], dtype=self.dtype
        )
        self.out = np.bitwise_xor(self.x, self.y)
        self.perm = [1, 0]
        self.y_trans = np.transpose(self.y, self.perm)

    def init_bound(self):
        self.low = -100
        self.high = 100

    def test_check_grad(self):
        pass


class TestElementwiseBitwiseXorOp_Stride1(TestElementwiseBitwiseXorOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.randint(
            self.low, self.high, [20, 2, 13, 17], dtype=self.dtype
        )
        self.y = np.random.randint(
            self.low, self.high, [20, 2, 13, 17], dtype=self.dtype
        )
        self.out = np.bitwise_xor(self.x, self.y)
        self.perm = [0, 1, 3, 2]
        self.y_trans = np.transpose(self.y, self.perm)


class TestElementwiseBitwiseXorOp_Stride2(TestElementwiseBitwiseXorOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.randint(
            self.low, self.high, [20, 2, 13, 17], dtype=self.dtype
        )
        self.y = np.random.randint(
            self.low, self.high, [20, 2, 13, 17], dtype=self.dtype
        )
        self.out = np.bitwise_xor(self.x, self.y)
        self.perm = [0, 2, 1, 3]
        self.y_trans = np.transpose(self.y, self.perm)


class TestElementwiseBitwiseXorOp_Stride3(TestElementwiseBitwiseXorOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.randint(
            self.low, self.high, [20, 2, 13, 17], dtype=self.dtype
        )
        self.y = np.random.randint(
            self.low, self.high, [20, 2, 13, 1], dtype=self.dtype
        )
        self.out = np.bitwise_xor(self.x, self.y)
        self.perm = [0, 1, 3, 2]
        self.y_trans = np.transpose(self.y, self.perm)


class TestElementwiseBitwiseXorOp_Stride4(TestElementwiseBitwiseXorOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.randint(
            self.low, self.high, [1, 2, 13, 17], dtype=self.dtype
        )
        self.y = np.random.randint(
            self.low, self.high, [20, 2, 13, 1], dtype=self.dtype
        )
        self.out = np.bitwise_xor(self.x, self.y)
        self.perm = [1, 0, 2, 3]
        self.y_trans = np.transpose(self.y, self.perm)


class TestElementwiseBitwiseXorOp_Stride5(TestElementwiseBitwiseXorOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "as_stride"
        self.x = np.random.randint(
            self.low, self.high, [23, 10, 1, 17], dtype=self.dtype
        )
        self.y = np.random.randint(
            self.low, self.high, [23, 2, 13, 20], dtype=self.dtype
        )
        self.y_trans = self.y
        self.y = self.y[:, 0:1, :, 0:1]
        self.out = np.bitwise_xor(self.x, self.y)
        self.shape_param = [23, 1, 13, 1]
        self.stride_param = [520, 260, 20, 1]


class TestElementwiseBitwiseXorOp_Stride_ZeroDim1(
    TestElementwiseBitwiseXorOp_Stride
):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.randint(self.low, self.high, [], dtype=self.dtype)
        self.y = np.random.randint(
            self.low, self.high, [13, 17], dtype=self.dtype
        )
        self.out = np.bitwise_xor(self.x, self.y)
        self.perm = [1, 0]
        self.y_trans = np.transpose(self.y, self.perm)


class TestElementwiseBitwiseXorOp_Stride_ZeroSize1(
    TestElementwiseBitwiseXorOp_Stride
):
    def init_data(self):
        self.strided_input_type = "transpose"
        self.x = np.random.rand(1, 0, 2).astype('float32')
        self.y = np.random.rand(3, 0, 1).astype('float32')
        self.out = np.bitwise_xor(self.x, self.y)
        self.perm = [2, 1, 0]
        self.y_trans = np.transpose(self.y, self.perm)


# ---------------  TEST OP: BitwiseNot ----------------- #
class TestBitwiseNot(OpTest):
    def setUp(self):
        self.op_type = "bitwise_not"
        self.python_api = paddle.tensor.logic.bitwise_not

        self.init_dtype()
        self.init_shape()
        self.init_bound()

        x = np.random.randint(
            self.low, self.high, self.x_shape, dtype=self.dtype
        )
        out = np.bitwise_not(x)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(
            check_cinn=True, check_pir=True, check_symbol_infer=False
        )

    def test_check_grad(self):
        pass

    def init_dtype(self):
        self.dtype = np.int32

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]

    def init_bound(self):
        self.low = -100
        self.high = 100


class TestBitwiseNot_ZeroDim(TestBitwiseNot):
    def init_shape(self):
        self.x_shape = []


class TestBitwiseNot_ZeroSize(TestBitwiseNot):
    def init_shape(self):
        self.x_shape = [0, 3, 4, 5]


class TestBitwiseNotUInt8(TestBitwiseNot):
    def init_dtype(self):
        self.dtype = np.uint8

    def init_bound(self):
        self.low = 0
        self.high = 100


class TestBitwiseNotInt8(TestBitwiseNot):
    def init_dtype(self):
        self.dtype = np.int8

    def init_shape(self):
        self.x_shape = [4, 5]


class TestBitwiseNotInt16(TestBitwiseNot):
    def init_dtype(self):
        self.dtype = np.int16

    def init_shape(self):
        self.x_shape = [2, 3, 4, 5]


class TestBitwiseNotInt64(TestBitwiseNot):
    def init_dtype(self):
        self.dtype = np.int64

    def init_shape(self):
        self.x_shape = [1, 4, 1]


class TestBitwiseNotBool(TestBitwiseNot):
    def setUp(self):
        self.op_type = "bitwise_not"
        self.python_api = paddle.tensor.logic.bitwise_not
        self.init_shape()

        x = np.random.choice([True, False], self.x_shape)
        out = np.bitwise_not(x)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}


class TestBitwiseInvertApi(unittest.TestCase):
    def setUp(self):
        paddle.disable_static()

        self.dtype = np.int32
        self.shape = [2, 3, 4, 5]
        self.low = -100
        self.high = 100
        x = np.random.randint(self.low, self.high, self.shape, dtype=self.dtype)
        self.x = paddle.to_tensor(x)
        self.expected_out = np.bitwise_not(x)

    def test_bitwise_invert_out_of_place(self):
        result = paddle.bitwise_invert(self.x)
        np.testing.assert_array_equal(result.numpy(), self.expected_out)

    def test_bitwise_invert_in_place(self):
        x_copy = self.x.clone()
        x_copy.bitwise_invert_()
        np.testing.assert_array_equal(x_copy.numpy(), self.expected_out)


if __name__ == "__main__":
    unittest.main()
