#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import op_test

import paddle
from paddle import base
from paddle.base import framework


def wrap_assign_value_wrapper(dtype=paddle.float32):
    def assign_value_wrapper(shape=[], dtype=dtype, values=0.0):
        if paddle.framework.in_dynamic_mode():
            tensor = paddle.Tensor()
        else:
            tensor = paddle.zeros(list(shape), dtype)
        return paddle._C_ops.assign_value_(
            tensor, shape, dtype, values, framework._current_expected_place()
        )

    return assign_value_wrapper


class TestAssignValueOp(op_test.OpTest):
    def setUp(self):
        self.op_type = "assign_value"
        self.inputs = {}
        self.attrs = {}
        self.init_data()
        self.attrs["shape"] = self.value.shape
        self.attrs["dtype"] = framework.convert_np_dtype_to_dtype_(
            self.value.dtype
        )
        self.python_api = wrap_assign_value_wrapper(self.attrs["dtype"])
        self.outputs = {"Out": self.value}

    def init_data(self):
        self.value = np.random.random(size=(2, 5)).astype(np.float32)
        self.attrs["values"] = [float(v) for v in self.value.flat]

    def test_forward(self):
        self.check_output(check_cinn=True, check_pir=True)


class TestAssignValueOp2(TestAssignValueOp):
    def init_data(self):
        self.value = np.random.random(size=(2, 5)).astype(np.int32)
        self.attrs["values"] = [int(v) for v in self.value.flat]


class TestAssignValueOp3(TestAssignValueOp):
    def init_data(self):
        self.value = np.random.random(size=(2, 5)).astype(np.int64)
        self.attrs["values"] = [int(v) for v in self.value.flat]


class TestAssignValueOp4(TestAssignValueOp):
    def init_data(self):
        self.value = np.random.choice(a=[False, True], size=(2, 5)).astype(
            np.bool_
        )
        self.attrs["values"] = [int(v) for v in self.value.flat]


class TestAssignValueOp5(TestAssignValueOp):
    def init_data(self):
        self.value = np.random.random(size=(2, 5)).astype(np.float64)
        self.attrs["values"] = [float(v) for v in self.value.flat]


class TestAssignValueOp6(TestAssignValueOp):
    def init_data(self):
        self.value = (
            np.random.random(size=(2, 5)) + 1j * np.random.random(size=(2, 5))
        ).astype(np.complex64)
        self.attrs["values"] = list(self.value.flat)


class TestAssignValueOp7(TestAssignValueOp):
    def init_data(self):
        self.value = (
            np.random.random(size=(2, 5)) + 1j * np.random.random(size=(2, 5))
        ).astype(np.complex128)
        self.attrs["values"] = list(self.value.flat)


class TestAssignApi(unittest.TestCase):
    def setUp(self):
        with op_test.paddle_static_guard():
            self.init_dtype()
            self.value = (-100 + 200 * np.random.random(size=(2, 5))).astype(
                self.dtype
            )
            self.place = (
                base.CUDAPlace(0)
                if base.is_compiled_with_cuda()
                else base.CPUPlace()
            )

    def init_dtype(self):
        self.dtype = "float32"

    def test_assign(self):
        with op_test.paddle_static_guard():
            main_program = base.Program()
            with base.program_guard(main_program):
                x = paddle.assign(self.value)

            exe = base.Executor(self.place)
            [fetched_x] = exe.run(main_program, feed={}, fetch_list=[x])
            np.testing.assert_array_equal(fetched_x, self.value)
            self.assertEqual(fetched_x.dtype, self.value.dtype)

    def test_pir_assign(self):
        with paddle.pir_utils.IrGuard():
            main_program = paddle.pir.Program()
            with paddle.static.program_guard(main_program):
                x = paddle.zeros(shape=[1], dtype=self.dtype)
                paddle.assign(self.value, output=x)

            exe = base.Executor(self.place)
            [fetched_x] = exe.run(main_program, feed={}, fetch_list=[x])
            np.testing.assert_array_equal(fetched_x, self.value)
            self.assertEqual(fetched_x.dtype, self.value.dtype)


class TestAssignApi2(TestAssignApi):
    def init_dtype(self):
        self.dtype = "int32"


class TestAssignApi3(TestAssignApi):
    def init_dtype(self):
        self.dtype = "int64"


class TestAssignApi4(TestAssignApi):
    def setUp(self):
        with op_test.paddle_static_guard():
            self.init_dtype()
            self.value = np.random.choice(a=[False, True], size=(2, 5)).astype(
                np.bool_
            )
            self.place = (
                base.CUDAPlace(0)
                if base.is_compiled_with_cuda()
                else base.CPUPlace()
            )

    def init_dtype(self):
        self.dtype = "bool"


class TestAssignApi5(TestAssignApi):
    def init_dtype(self):
        self.dtype = "float64"


class TestAssignApi6(TestAssignApi):
    def setUp(self):
        with op_test.paddle_static_guard():
            self.init_dtype()
            self.value = (
                np.random.random(size=(2, 5))
                + 1j * (np.random.random(size=(2, 5)))
            ).astype(np.complex64)
            self.place = (
                base.CUDAPlace(0)
                if base.is_compiled_with_cuda()
                else base.CPUPlace()
            )

    def init_dtype(self):
        self.dtype = "complex64"


class TestAssignApi7(TestAssignApi):
    def setUp(self):
        with op_test.paddle_static_guard():
            self.init_dtype()
            self.value = (
                np.random.random(size=(2, 5))
                + 1j * (np.random.random(size=(2, 5)))
            ).astype(np.complex128)
            self.place = (
                base.CUDAPlace(0)
                if base.is_compiled_with_cuda()
                else base.CPUPlace()
            )

    def init_dtype(self):
        self.dtype = "complex128"


if __name__ == '__main__':
    unittest.main()
