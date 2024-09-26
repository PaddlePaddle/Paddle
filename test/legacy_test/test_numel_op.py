#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import OpTest, convert_float_to_uint16

import paddle
from paddle.base import core


class TestNumelOp(OpTest):
    def setUp(self):
        self.op_type = "size"
        self.prim_op_type = "comp"
        self.python_api = paddle.numel
        self.public_python_api = paddle.numel
        self.init()
        x = np.random.random(self.shape).astype(self.dtype)
        self.inputs = {
            'Input': x,
        }
        self.outputs = {'Out': np.array(np.size(x))}

    def test_check_output(self):
        self.check_output(check_pir=True, check_prim_pir=True)

    def init(self):
        self.shape = (6, 56, 8, 55)
        self.dtype = np.float64


class TestNumelOp1(TestNumelOp):
    def init(self):
        self.shape = (11, 66)
        self.dtype = np.float64


class TestNumelOp2(TestNumelOp):
    def init(self):
        self.shape = (0,)
        self.dtype = np.float64


class TestNumelOpFP16(TestNumelOp):
    def init(self):
        self.dtype = np.float16
        self.shape = (6, 56, 8, 55)


class TestNumelOp1FP16(TestNumelOp):
    def init(self):
        self.dtype = np.float16
        self.shape = (11, 66)


class TestNumelOp2FP16(TestNumelOp):
    def init(self):
        self.dtype = np.float16
        self.shape = (0,)


class TestNumelOp1int8(TestNumelOp):
    def init(self):
        self.dtype = np.int8
        self.shape = (11, 66)


class TestNumelOp2int8(TestNumelOp):
    def init(self):
        self.dtype = np.int8
        self.shape = (0,)


class TestNumelOpComplex(TestNumelOp):
    def setUp(self):
        self.op_type = "size"
        self.prim_op_type = "comp"
        self.python_api = paddle.numel
        self.public_python_api = paddle.numel
        self.init()
        x = np.random.random(self.shape).astype(
            self.dtype
        ) + 1j * np.random.random(self.shape).astype(self.dtype)
        self.inputs = {
            'Input': x,
        }
        self.outputs = {'Out': np.array(np.size(x))}

    def init(self):
        self.dtype = np.complex64
        self.shape = (6, 56, 8, 55)


class TestNumelOp1Complex64(TestNumelOpComplex):
    def init(self):
        self.dtype = np.complex64
        self.shape = (11, 66)


class TestNumelOp2Complex64(TestNumelOpComplex):
    def init(self):
        self.dtype = np.complex64
        self.shape = (0,)


class TestNumelOp0Complex128(TestNumelOpComplex):
    def init(self):
        self.dtype = np.complex128
        self.shape = (6, 56, 8, 55)


class TestNumelOp1Complex128(TestNumelOpComplex):
    def init(self):
        self.dtype = np.complex128
        self.shape = (11, 66)


class TestNumelOp2Complex128(TestNumelOpComplex):
    def init(self):
        self.dtype = np.complex128
        self.shape = (0,)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and do not support bfloat16",
)
class TestNumelOpBF16(OpTest):
    def setUp(self):
        self.op_type = "size"
        self.prim_op_type = "comp"
        self.python_api = paddle.numel
        self.public_python_api = paddle.numel
        self.dtype = np.uint16
        self.init()
        x = np.random.random(self.shape).astype(np.float32)
        self.inputs = {'Input': convert_float_to_uint16(x)}
        self.outputs = {'Out': np.array(np.size(x))}

    def test_check_output(self):
        place = paddle.CUDAPlace(0)
        self.check_output_with_place(place, check_pir=True, check_prim_pir=True)

    def init(self):
        self.shape = (6, 56, 8, 55)


class TestNumelOp1BF16(TestNumelOpBF16):
    def init(self):
        self.shape = (11, 66)


class TestNumelAPI(unittest.TestCase):

    def test_numel_static(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):
            shape1 = [2, 1, 4, 5]
            shape2 = [1, 4, 5]
            x_1 = paddle.static.data(shape=shape1, dtype='int32', name='x_1')
            x_2 = paddle.static.data(shape=shape2, dtype='int32', name='x_2')
            input_1 = np.random.random(shape1).astype("int32")
            input_2 = np.random.random(shape2).astype("int32")
            out_1 = paddle.numel(x_1)
            out_2 = paddle.numel(x_2)
            exe = paddle.static.Executor(place=paddle.CPUPlace())
            res_1, res_2 = exe.run(
                feed={
                    "x_1": input_1,
                    "x_2": input_2,
                },
                fetch_list=[out_1, out_2],
            )
            np.testing.assert_array_equal(
                res_1, np.array(np.size(input_1)).astype("int64")
            )
            np.testing.assert_array_equal(
                res_2, np.array(np.size(input_2)).astype("int64")
            )

    def test_numel_imperative(self):
        paddle.disable_static(paddle.CPUPlace())
        input_1 = np.random.random([2, 1, 4, 5]).astype("int32")
        input_2 = np.random.random([1, 4, 5]).astype("int32")
        x_1 = paddle.to_tensor(input_1)
        x_2 = paddle.to_tensor(input_2)
        out_1 = paddle.numel(x_1)
        out_2 = paddle.numel(x_2)
        np.testing.assert_array_equal(out_1.numpy().item(0), np.size(input_1))
        np.testing.assert_array_equal(out_2.numpy().item(0), np.size(input_2))
        paddle.enable_static()

    def test_error(self):
        main_program = paddle.static.Program()
        startup_program = paddle.static.Program()
        with paddle.static.program_guard(main_program, startup_program):

            def test_x_type():
                shape = [1, 4, 5]
                input_1 = np.random.random(shape).astype("int32")
                out_1 = paddle.numel(input_1)

            self.assertRaises(TypeError, test_x_type)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
