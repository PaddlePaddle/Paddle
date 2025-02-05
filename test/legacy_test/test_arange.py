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
from paddle.static import Program, program_guard


def arange_wrapper(start, end, step, dtype="float32"):
    return paddle.arange(start, end, step, dtype)


class TestArangeOp(OpTest):
    def setUp(self):
        self.op_type = "range"
        self.init_config()
        self.inputs = {
            'Start': np.array([self.case[0]]).astype(self.dtype),
            'End': np.array([self.case[1]]).astype(self.dtype),
            'Step': np.array([self.case[2]]).astype(self.dtype),
        }

        self.outputs = {
            'Out': np.arange(self.case[0], self.case[1], self.case[2]).astype(
                self.dtype
            )
        }

    def init_config(self):
        self.dtype = np.float32
        self.python_api = arange_wrapper
        self.case = (0, 1, 0.2)

    def test_check_output(self):
        self.check_output(check_pir=True, check_symbol_infer=False)


class TestFloatArangeOp(TestArangeOp):
    def init_config(self):
        self.dtype = np.float32
        self.python_api = paddle.arange
        self.case = (0, 5, 1)


class TestFloat16ArangeOp(TestArangeOp):
    def init_config(self):
        self.dtype = np.float16
        self.python_api = paddle.arange
        self.case = (0, 5, 1)


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestBFloat16ArangeOp(OpTest):
    def setUp(self):
        self.op_type = "range"
        self.init_config()
        self.inputs = {
            'Start': convert_float_to_uint16(self.start),
            'End': convert_float_to_uint16(self.end),
            'Step': convert_float_to_uint16(self.step),
        }

        self.outputs = {
            'Out': convert_float_to_uint16(
                np.arange(self.start, self.end, self.step)
            )
        }

    def init_config(self):
        self.dtype = np.uint16
        self.python_api = arange_wrapper
        self.case = (0, 5, 1)
        self.start = np.array([self.case[0]]).astype(np.float32)
        self.end = np.array([self.case[1]]).astype(np.float32)
        self.step = np.array([self.case[2]]).astype(np.float32)

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(
            place, check_pir=True, check_symbol_infer=False
        )


class TestInt32ArangeOp(TestArangeOp):
    def init_config(self):
        self.dtype = np.int32
        self.python_api = paddle.arange
        self.case = (0, 5, 2)


class TestFloat64ArangeOp(TestArangeOp):
    def init_config(self):
        self.dtype = np.float64
        self.python_api = paddle.arange
        self.case = (10, 1, -2)


class TestInt64ArangeOp(TestArangeOp):
    def init_config(self):
        self.dtype = np.int64
        self.python_api = paddle.arange
        self.case = (-1, -10, -2)


class TestZeroSizeArangeOp(TestArangeOp):
    def init_config(self):
        self.dtype = np.int32
        self.python_api = paddle.arange
        self.case = (0, 0, 1)


class TestArangeOpError(unittest.TestCase):

    def test_static_errors(self):
        with program_guard(Program(), Program()):
            paddle.enable_static()
            self.assertRaises(TypeError, paddle.arange, 10, dtype='int8')


class TestArangeAPI(unittest.TestCase):

    def test_out(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            x1 = paddle.arange(0, 5, 1, 'float32')

            place = (
                paddle.CUDAPlace(0)
                if core.is_compiled_with_cuda()
                else paddle.CPUPlace()
            )
            exe = paddle.static.Executor(place)
            out = exe.run(fetch_list=[x1])

            expected_data = np.arange(0, 5, 1).astype(np.float32)
            self.assertEqual((out == expected_data).all(), True)
            self.assertListEqual(list(x1.shape), [5])
        paddle.disable_static(place)


class TestArangeImperative(unittest.TestCase):
    def test_out(self):
        place = (
            paddle.CUDAPlace(0)
            if core.is_compiled_with_cuda()
            else paddle.CPUPlace()
        )
        paddle.disable_static(place)
        x1 = paddle.arange(0, 5, 1)
        x2 = paddle.tensor.arange(5)
        x3 = paddle.tensor.creation.arange(5)

        start = paddle.to_tensor(np.array([0], 'float32'))
        end = paddle.to_tensor(np.array([5], 'float32'))
        step = paddle.to_tensor(np.array([1], 'float32'))
        x4 = paddle.arange(start, end, step, 'int64')

        expected_data = np.arange(0, 5, 1).astype(np.int64)
        for x in [x1, x2, x3, x4]:
            np.testing.assert_array_equal(x.numpy(), expected_data)

        start_float = paddle.to_tensor(np.array([0.5], 'float32'))
        end_float = paddle.to_tensor(np.array([1.5], 'float32'))
        step_float = paddle.to_tensor(np.array([0.5], 'float32'))
        # all [start, end, step] is float
        x5 = paddle.arange(start_float, end_float, step_float)
        x5_expected_data = np.arange(0.5, 1.5, 0.5).astype(np.float32)
        np.testing.assert_array_equal(x5.numpy(), x5_expected_data)
        self.assertEqual(x5.numpy().dtype, np.float32)

        # [start, end] is float , [step] is int
        x6 = paddle.arange(start_float, end_float, 1)
        x6_expected_data = np.arange(0.5, 1.5, 1).astype(np.float32)
        np.testing.assert_array_equal(x6.numpy(), x6_expected_data)
        self.assertEqual(x6.numpy().dtype, np.float32)

        # [start] is float , [end] is int
        x7 = paddle.arange(start_float, 1)
        x7_expected_data = np.arange(0.5, 1).astype(np.float32)
        np.testing.assert_array_equal(x7.numpy(), x7_expected_data)
        self.assertEqual(x7.numpy().dtype, np.float32)

        # [start] is float
        x8 = paddle.arange(start_float)
        x8_expected_data = np.arange(0.5).astype(np.float32)
        np.testing.assert_array_equal(x8.numpy(), x8_expected_data)
        self.assertEqual(x8.numpy().dtype, np.float32)

        # [start] is int
        x9 = paddle.arange(1)
        x9_expected_data = np.arange(1).astype(np.int64)
        np.testing.assert_array_equal(x9.numpy(), x9_expected_data)
        self.assertEqual(x9.numpy().dtype, np.int64)

        # [start] is float
        x10 = paddle.arange(1.0)
        x10_expected_data = np.arange(1).astype(np.float32)
        np.testing.assert_array_equal(x10.numpy(), x10_expected_data)
        self.assertEqual(x10.numpy().dtype, np.float32)

        # [start] is np.int
        x11 = paddle.arange(np.int64(10))
        x11_expected_data = np.arange(10).astype(np.int64)
        np.testing.assert_array_equal(x11.numpy(), x11_expected_data)
        self.assertEqual(x11.numpy().dtype, np.int64)

        # [start] is a big integer
        x12 = paddle.arange(
            start=0,
            end=-9007199254740994,
            step=-9007199254740993,
        )

        # numpy give wrong result here, so we generate 'x12_expected_data' manually
        # x12_expected_data = np.arange(start=0, stop=-9007199254740994, step=-9007199254740993, dtype=np.int64)
        x12_expected_data = np.array([0, -9007199254740993])

        np.testing.assert_array_equal(x12.numpy(), x12_expected_data)
        self.assertEqual(x12.numpy().dtype, np.int64)

        paddle.enable_static()


class TestArangeStatic(unittest.TestCase):
    def test_infermeta(self):
        paddle.enable_static()
        x = paddle.arange(0, 1 + 0.005, 0.005)
        self.assertEqual(x.shape, [201])
        paddle.disable_static()


if __name__ == "__main__":
    unittest.main()
