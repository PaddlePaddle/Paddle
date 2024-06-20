#   Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle.static import Program, program_guard

# Support types are ref from `paddle.tensor.math`
# - Related paddle dtypes:
#  - int type: int64, uint8, int8, int16, int32
#  - float type: float16, bfloat16, float32, float64
#  - complex type: complex64, complex128
# - Python scalar dtypes:
#  - int(64)
#  - float(32)
#  - complex(64)


class TestTensorScalarTypePromotionStatic(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()

    def check_operation(self, a, b, c, op):
        exe = paddle.static.Executor()

        if op == '+':
            c_rlt = a + b
        elif op == '-':
            c_rlt = a - b
        elif op == '*':
            c_rlt = a * b
        elif op == '/':
            c_rlt = a / b
        elif op == '**':
            c_rlt = a**b
        elif op == '//':
            c_rlt = a // b
        elif op == '%':
            c_rlt = a % b
        else:
            raise ValueError("Unsupported operation.")

        rlt = exe.run(fetch_list=[c_rlt, c])

        self.assertEqual(rlt[0].dtype, rlt[1].dtype)
        np.testing.assert_array_equal(rlt[0], rlt[1])

    def test_tensor_add_scalar(self):
        # tensor(int64) + scalar(int)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 1
            c = paddle.full([2, 2, 2], 2, dtype="int64")
            self.check_operation(a, b, c, '+')

        # tensor(float32) + scalar(int)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='float32')
            b = 1
            c = paddle.full([2, 2, 2], 2, dtype="float32")
            self.check_operation(a, b, c, '+')

        # tensor(int64) + scalar(float, .0)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 1.0
            c = paddle.full([2, 2, 2], 2, dtype="float32")
            self.check_operation(a, b, c, '+')

        # tensor(int64) + scalar(float, .5)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 1.5
            c = paddle.full([2, 2, 2], 2.5, dtype="float32")
            self.check_operation(a, b, c, '+')

        # tensor(float32) + scalar(float)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='float32')
            b = 1.5
            c = paddle.full([2, 2, 2], 2.5, dtype="float32")
            self.check_operation(a, b, c, '+')

    def test_tensor_sub_scalar(self):
        # tensor(int64) - scalar(int)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 1
            c = paddle.zeros([2, 2, 2], dtype="int64")
            self.check_operation(a, b, c, '-')

        # tensor(float32) - scalar(int)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='float32')
            b = 1
            c = paddle.zeros([2, 2, 2], dtype="float32")
            self.check_operation(a, b, c, '-')

        # tensor(int64) - scalar(float, .0)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 1.0
            c = paddle.zeros([2, 2, 2], dtype="float32")
            self.check_operation(a, b, c, '-')

        # tensor(int64) - scalar(float, .5)
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 2, dtype='int64')
            b = 1.5
            c = paddle.full([2, 2, 2], 0.5, dtype="float32")
            self.check_operation(a, b, c, '-')

        # tensor(float32) - scalar(float)
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 2, dtype='float32')
            b = 1.5
            c = paddle.full([2, 2, 2], 0.5, dtype="float32")
            self.check_operation(a, b, c, '-')

    def test_scalar_sub_tensor(self):
        # scalar(int) - tensor(int64)
        with program_guard(Program()):
            a = 1
            b = paddle.ones([2, 2, 2], dtype='int64')
            c = paddle.zeros([2, 2, 2], dtype="int64")
            self.check_operation(a, b, c, '-')

        # scalar(int) - tensor(float32)
        with program_guard(Program()):
            a = 1
            b = paddle.ones([2, 2, 2], dtype='float32')
            c = paddle.zeros([2, 2, 2], dtype="float32")
            self.check_operation(a, b, c, '-')

        # scalar(float, .0) - tensor(int64)
        with program_guard(Program()):
            a = 1.0
            b = paddle.ones([2, 2, 2], dtype='int64')
            c = paddle.zeros([2, 2, 2], dtype="float32")
            self.check_operation(a, b, c, '-')

        # scalar(float, .5) - tensor(int64)
        with program_guard(Program()):
            a = 1.5
            b = paddle.full([2, 2, 2], 2, dtype='int64')
            c = paddle.full([2, 2, 2], -0.5, dtype="float32")
            self.check_operation(a, b, c, '-')

        # scalar(float) - tensor(float32)
        with program_guard(Program()):
            a = 1.5
            b = paddle.full([2, 2, 2], 2, dtype='float32')
            c = paddle.full([2, 2, 2], -0.5, dtype="float32")
            self.check_operation(a, b, c, '-')

    def test_tensor_mul_tensor(self):
        # tensor(int64) * scalar(int)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 1
            c = paddle.ones([2, 2, 2], dtype="int64")
            self.check_operation(a, b, c, '*')

        # tensor(float32) * scalar(int)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='float32')
            b = 1
            c = paddle.ones([2, 2, 2], dtype="float32")
            self.check_operation(a, b, c, '*')

        # tensor(int64) * scalar(float, .0)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 1.0
            c = paddle.ones([2, 2, 2], dtype="float32")
            self.check_operation(a, b, c, '*')

        # tensor(int64) * scalar(float, .5)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 1.5
            c = paddle.full([2, 2, 2], 1.5, dtype="float32")
            self.check_operation(a, b, c, '*')

        # tensor(float32) * scalar(float)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='float32')
            b = 1.5
            c = paddle.full([2, 2, 2], 1.5, dtype="float32")
            self.check_operation(a, b, c, '*')

    def test_tensor_div_scalar(self):
        # tensor(int64) / scalar(int)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 2
            c = paddle.full([2, 2, 2], 0.5, dtype="float32")
            self.check_operation(a, b, c, '/')

        # tensor(float32) / scalar(int)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='float32')
            b = 2
            c = paddle.full([2, 2, 2], 0.5, dtype="float32")
            self.check_operation(a, b, c, '/')

        # tensor(int64) / scalar(float, .0)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 2.0
            c = paddle.full([2, 2, 2], 0.5, dtype="float32")
            self.check_operation(a, b, c, '/')

        # tensor(int64) / scalar(float, .5)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='int64')
            b = 0.5
            c = paddle.full([2, 2, 2], 2, dtype="float32")
            self.check_operation(a, b, c, '/')

        # tensor(float32) / scalar(float)
        with program_guard(Program()):
            a = paddle.ones([2, 2, 2], dtype='float32')
            b = 0.5
            c = paddle.full([2, 2, 2], 2, dtype="float32")
            self.check_operation(a, b, c, '/')

    def test_scalar_div_tensor(self):
        # scalar(int) / tensor(int64)
        with program_guard(Program()):
            a = 1
            b = paddle.full([2, 2, 2], 2, dtype='int64')
            c = paddle.full([2, 2, 2], 0.5, dtype="float32")
            self.check_operation(a, b, c, '/')

        # scalar(int) / tensor(float32)
        with program_guard(Program()):
            a = 1
            b = paddle.full([2, 2, 2], 0.5, dtype='float32')
            c = paddle.full([2, 2, 2], 2, dtype="float32")
            self.check_operation(a, b, c, '/')

        # scalar(float) / tensor(int64)
        with program_guard(Program()):
            a = 1.0
            b = paddle.full([2, 2, 2], 2, dtype='int64')
            c = paddle.full([2, 2, 2], 0.5, dtype="float32")
            self.check_operation(a, b, c, '/')

        # scalar(float) / tensor(float32)
        with program_guard(Program()):
            a = 1.0
            b = paddle.full([2, 2, 2], 0.5, dtype='float32')
            c = paddle.full([2, 2, 2], 2, dtype="float32")
            self.check_operation(a, b, c, '/')

    def test_tensor_pow_scalar(self):
        # tensor(int64) ** scalar(int)
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 2, dtype='int64')
            b = 3
            c = paddle.full([2, 2, 2], 8, dtype="int64")
            self.check_operation(a, b, c, '**')

        # tensor(int64) ** scalar(float)
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 2, dtype='int64')
            b = 3.0
            c = paddle.full([2, 2, 2], 8, dtype="float32")
            self.check_operation(a, b, c, '**')

        # tensor(float32) ** scalar(int)
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 2, dtype='float32')
            b = 3
            c = paddle.full([2, 2, 2], 8, dtype="float32")
            self.check_operation(a, b, c, '**')

        # tensor(float32) ** scalar(float)
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 2, dtype='float32')
            b = 3.0
            c = paddle.full([2, 2, 2], 8, dtype="float32")
            self.check_operation(a, b, c, '**')

    def test_scalar_pow_tensor(self):
        # scalar(int) ** tensor(int64)
        with program_guard(Program()):
            a = 3
            b = paddle.full([2, 2, 2], 2, dtype='int64')
            c = paddle.full([2, 2, 2], 9, dtype="int64")
            self.check_operation(a, b, c, '**')

        # scalar(float) ** tensor(int64)
        with program_guard(Program()):
            a = 3.0
            b = paddle.full([2, 2, 2], 2, dtype='int64')
            c = paddle.full([2, 2, 2], 9, dtype="float32")
            self.check_operation(a, b, c, '**')

        # scalar(int) ** tensor(float32)
        with program_guard(Program()):
            a = 3
            b = paddle.full([2, 2, 2], 2, dtype='float32')
            c = paddle.full([2, 2, 2], 9, dtype="float32")
            self.check_operation(a, b, c, '**')

        # tensor(float32) ** scalar(float)
        with program_guard(Program()):
            a = 3.0
            b = paddle.full([2, 2, 2], 2, dtype='float32')
            c = paddle.full([2, 2, 2], 9, dtype="float32")
            self.check_operation(a, b, c, '**')

    # ## TODO: floordiv op kernel doesn't support float
    def test_tensor_floordiv_scalar(self):
        # tensor(int64) // scalar(int)
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 3, dtype='int64')
            b = 2
            c = paddle.full([2, 2, 2], 1, dtype="int64")
            self.check_operation(a, b, c, '//')

    def test_tensor_mod_scalar(self):
        # tensor(int64) % scalar(int)
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 3, dtype='int64')
            b = 2
            c = paddle.full([2, 2, 2], 1, dtype="int64")
            self.check_operation(a, b, c, '%')

        # tensor(int64) % scalar(float)
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 3, dtype='int64')
            b = 2.0
            c = paddle.full([2, 2, 2], 1, dtype="float32")
            self.check_operation(a, b, c, '%')

        # tensor(float32) % scalar(int)
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 3, dtype='float32')
            b = 2
            c = paddle.full([2, 2, 2], 1, dtype="float32")
            self.check_operation(a, b, c, '%')

        # tensor(float32) % scalar(float)
        with program_guard(Program()):
            a = paddle.full([2, 2, 2], 3, dtype='float32')
            b = 2.0
            c = paddle.full([2, 2, 2], 1, dtype="float32")
            self.check_operation(a, b, c, '%')


def create_test_case(
    baseclass,
    dtype,
    expected_out_dtype_with_int=None,
    expected_out_dtype_with_float=None,
    expected_out_dtype_with_complex=None,
):
    class TestPromotion(baseclass):
        def set_dtype(self):
            self.dtype = dtype
            self.expected_out_dtype_with_int = expected_out_dtype_with_int
            self.expected_out_dtype_with_float = expected_out_dtype_with_float
            # static not support compute with scalar complex
            self.expected_out_dtype_with_complex = (
                expected_out_dtype_with_complex
            )

    cls_name = f"{baseclass.__name__}{dtype}"
    TestPromotion.__name__ = cls_name
    globals()[cls_name] = TestPromotion


class TestTensorAddScalar(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.set_dtype()
        self.exe = paddle.static.Executor()

    def set_dtype(self):
        self.dtype = 'float32'
        self.expected_out_dtype_with_int = 'float32'
        self.expected_out_dtype_with_float = 'float32'
        self.expected_out_dtype_with_complex = 'complex64'

    def generate_test_value(self):
        self.value = paddle.rand([2, 3, 4]).astype(self.dtype)

    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()
            out_int = self.value + 1
            out_float = self.value + 1.0
            # out_complex = self.value + 2j
            res = self.exe.run(prog, fetch_list=[out_int, out_float])
        return res

    def test_dtype_is_expected(self):
        res_int, res_float = self.run_api()
        self.assertEqual(
            res_int.dtype.__str__(), self.expected_out_dtype_with_int
        )
        self.assertEqual(
            res_float.dtype.__str__(), self.expected_out_dtype_with_float
        )
        # self.assertEqual(res_complex.dtype.__str__(), self.expected_out_dtype_with_complex)


create_test_case(TestTensorAddScalar, 'bool', 'int64', 'float32', 'complex64')
create_test_case(TestTensorAddScalar, 'uint8', 'uint8', 'float32', 'complex64')
create_test_case(TestTensorAddScalar, 'int8', 'int8', 'float32', 'complex64')
create_test_case(TestTensorAddScalar, 'int32', 'int32', 'float32', 'complex64')
create_test_case(TestTensorAddScalar, 'int64', 'int64', 'float32', 'complex64')
create_test_case(
    TestTensorAddScalar, 'float16', 'float16', 'float16', 'complex64'
)
create_test_case(
    TestTensorAddScalar, 'bfloat16', 'uint16', 'uint16', 'complex64'
)
create_test_case(
    TestTensorAddScalar, 'float64', 'float64', 'float64', 'complex128'
)
create_test_case(
    TestTensorAddScalar, 'complex64', 'complex64', 'complex64', 'complex64'
)
create_test_case(
    TestTensorAddScalar, 'complex128', 'complex128', 'complex128', 'complex128'
)


class TestTensorSubScalar(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.set_dtype()
        self.exe = paddle.static.Executor()

    def set_dtype(self):
        self.dtype = 'float32'
        self.expected_out_dtype_with_int = 'float32'
        self.expected_out_dtype_with_float = 'float32'
        self.expected_out_dtype_with_complex = 'complex64'

    def generate_test_value(self):
        self.value = paddle.rand([2, 3, 4]).astype(self.dtype)

    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()
            out_int = self.value - 1
            out_float = self.value - 1.0
            # out_complex = self.value - 2j
            res = self.exe.run(prog, fetch_list=[out_int, out_float])
        return res

    def test_dtype_is_expected(self):
        res_int, res_float = self.run_api()
        self.assertEqual(
            res_int.dtype.__str__(), self.expected_out_dtype_with_int
        )
        self.assertEqual(
            res_float.dtype.__str__(), self.expected_out_dtype_with_float
        )
        # self.assertEqual(res_complex.dtype.__str__(), self.expected_out_dtype_with_complex)


create_test_case(TestTensorSubScalar, 'bool', 'int64', 'float32', 'complex64')
create_test_case(TestTensorSubScalar, 'uint8', 'uint8', 'float32', 'complex64')
create_test_case(TestTensorSubScalar, 'int8', 'int8', 'float32', 'complex64')
create_test_case(TestTensorSubScalar, 'int32', 'int32', 'float32', 'complex64')
create_test_case(TestTensorSubScalar, 'int64', 'int64', 'float32', 'complex64')
create_test_case(
    TestTensorSubScalar, 'float16', 'float16', 'float16', 'complex64'
)
create_test_case(
    TestTensorSubScalar, 'bfloat16', 'uint16', 'uint16', 'complex64'
)
create_test_case(
    TestTensorSubScalar, 'float64', 'float64', 'float64', 'complex128'
)
create_test_case(
    TestTensorSubScalar, 'complex64', 'complex64', 'complex64', 'complex64'
)
create_test_case(
    TestTensorSubScalar, 'complex128', 'complex128', 'complex128', 'complex128'
)


class TestTensorDivScalar(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.set_dtype()
        self.exe = paddle.static.Executor()

    def set_dtype(self):
        self.dtype = 'float32'
        self.expected_out_dtype_with_int = 'float32'
        self.expected_out_dtype_with_float = 'float32'
        self.expected_out_dtype_with_complex = 'complex64'

    def generate_test_value(self):
        self.value = paddle.rand([2, 3, 4]).astype(self.dtype)

    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()
            out_int = self.value / 1
            out_float = self.value / 1.0
            # out_complex = self.value / 2j
            res = self.exe.run(prog, fetch_list=[out_int, out_float])
        return res

    def test_dtype_is_expected(self):
        res_int, res_float = self.run_api()
        self.assertEqual(
            res_int.dtype.__str__(), self.expected_out_dtype_with_int
        )
        self.assertEqual(
            res_float.dtype.__str__(), self.expected_out_dtype_with_float
        )
        # self.assertEqual(res_complex.dtype.__str__(), self.expected_out_dtype_with_complex)


create_test_case(TestTensorDivScalar, 'bool', 'float32', 'float32', 'complex64')
create_test_case(
    TestTensorDivScalar, 'uint8', 'float32', 'float32', 'complex64'
)
create_test_case(TestTensorDivScalar, 'int8', 'float32', 'float32', 'complex64')
create_test_case(
    TestTensorDivScalar, 'int32', 'float32', 'float32', 'complex64'
)
create_test_case(
    TestTensorDivScalar, 'int64', 'float32', 'float32', 'complex64'
)
create_test_case(
    TestTensorDivScalar, 'float16', 'float16', 'float16', 'complex64'
)
create_test_case(
    TestTensorDivScalar, 'bfloat16', 'uint16', 'uint16', 'complex64'
)
create_test_case(
    TestTensorDivScalar, 'float64', 'float64', 'float64', 'complex128'
)
create_test_case(
    TestTensorDivScalar, 'complex64', 'complex64', 'complex64', 'complex64'
)
create_test_case(
    TestTensorDivScalar, 'complex128', 'complex128', 'complex128', 'complex128'
)


class TestTensorMulScalar(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.set_dtype()
        self.exe = paddle.static.Executor()

    def set_dtype(self):
        self.dtype = 'float32'
        self.expected_out_dtype_with_int = 'float32'
        self.expected_out_dtype_with_float = 'float32'
        self.expected_out_dtype_with_complex = 'complex64'

    def generate_test_value(self):
        self.value = paddle.rand([2, 3, 4]).astype(self.dtype)

    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()
            out_int = self.value * 1
            out_float = self.value * 1.0
            # out_complex = self.value * 2j
            res = self.exe.run(prog, fetch_list=[out_int, out_float])
        return res

    def test_dtype_is_expected(self):
        res_int, res_float = self.run_api()
        self.assertEqual(
            res_int.dtype.__str__(), self.expected_out_dtype_with_int
        )
        self.assertEqual(
            res_float.dtype.__str__(), self.expected_out_dtype_with_float
        )
        # self.assertEqual(res_complex.dtype.__str__(), self.expected_out_dtype_with_complex)


create_test_case(TestTensorMulScalar, 'bool', 'int64', 'float32', 'complex64')
create_test_case(TestTensorMulScalar, 'uint8', 'uint8', 'float32', 'complex64')
create_test_case(TestTensorMulScalar, 'int8', 'int8', 'float32', 'complex64')
create_test_case(TestTensorMulScalar, 'int32', 'int32', 'float32', 'complex64')
create_test_case(TestTensorMulScalar, 'int64', 'int64', 'float32', 'complex64')
create_test_case(
    TestTensorMulScalar, 'float16', 'float16', 'float16', 'complex64'
)
create_test_case(
    TestTensorMulScalar, 'bfloat16', 'uint16', 'uint16', 'complex64'
)
create_test_case(
    TestTensorMulScalar, 'float64', 'float64', 'float64', 'complex128'
)
create_test_case(
    TestTensorMulScalar, 'complex64', 'complex64', 'complex64', 'complex64'
)
create_test_case(
    TestTensorMulScalar, 'complex128', 'complex128', 'complex128', 'complex128'
)


class TestTensorPowScalar(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.set_dtype()
        self.exe = paddle.static.Executor()

    def set_dtype(self):
        self.dtype = 'float32'
        self.expected_out_dtype_with_int = 'float32'
        self.expected_out_dtype_with_float = 'float32'
        self.expected_out_dtype_with_complex = 'complex64'

    def generate_test_value(self):
        self.value = paddle.rand([2, 3, 4]).astype(self.dtype)

    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()
            out_int = self.value**1
            out_float = self.value**1.0
            # pow API not support complex
            res = self.exe.run(prog, fetch_list=[out_int, out_float])
        return res

    def test_dtype_is_expected(self):
        res_int, res_float = self.run_api()
        self.assertEqual(
            res_int.dtype.__str__(), self.expected_out_dtype_with_int
        )
        self.assertEqual(
            res_float.dtype.__str__(), self.expected_out_dtype_with_float
        )
        # self.assertEqual(res_complex.dtype.__str__(), self.expected_out_dtype_with_complex)


# pow API support int32, int64, float64, float32.
create_test_case(TestTensorPowScalar, 'int32', 'int32', 'float32', 'complex64')
create_test_case(TestTensorPowScalar, 'int64', 'int64', 'float32', 'complex64')
create_test_case(
    TestTensorPowScalar, 'float64', 'float64', 'float64', 'complex128'
)


class TestTensorFloorDivScalar(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.set_dtype()
        self.exe = paddle.static.Executor()

    def set_dtype(self):
        self.dtype = 'float32'
        self.expected_out_dtype_with_int = 'float32'
        self.expected_out_dtype_with_float = 'float32'
        self.expected_out_dtype_with_complex = 'complex64'

    def generate_test_value(self):
        self.value = paddle.rand([2, 3, 4]).astype(self.dtype)

    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()
            out_int = self.value // 1
            out_float = self.value // 1.0
            # floor_div API not support complex
            res = self.exe.run(prog, fetch_list=[out_int, out_float])
        return res

    def test_dtype_is_expected(self):
        res_int, res_float = self.run_api()
        self.assertEqual(
            res_int.dtype.__str__(), self.expected_out_dtype_with_int
        )
        self.assertEqual(
            res_float.dtype.__str__(), self.expected_out_dtype_with_float
        )
        # self.assertEqual(res_complex.dtype.__str__(), self.expected_out_dtype_with_complex)


# floor_div API not support complex64, complex128
create_test_case(
    TestTensorFloorDivScalar, 'bool', 'int64', 'float32', 'complex64'
)
create_test_case(
    TestTensorFloorDivScalar, 'uint8', 'uint8', 'float32', 'complex64'
)
create_test_case(
    TestTensorFloorDivScalar, 'int8', 'int8', 'float32', 'complex64'
)
create_test_case(
    TestTensorFloorDivScalar, 'int32', 'int32', 'float32', 'complex64'
)
create_test_case(
    TestTensorFloorDivScalar, 'int64', 'int64', 'float32', 'complex64'
)
create_test_case(
    TestTensorFloorDivScalar, 'float16', 'float16', 'float16', 'complex64'
)
create_test_case(
    TestTensorFloorDivScalar, 'bfloat16', 'uint16', 'uint16', 'complex64'
)
create_test_case(
    TestTensorFloorDivScalar, 'float64', 'float64', 'float64', 'complex128'
)


class TestTensorModScalar(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        self.set_dtype()
        self.exe = paddle.static.Executor()

    def set_dtype(self):
        self.dtype = 'float32'
        self.expected_out_dtype_with_int = 'float32'
        self.expected_out_dtype_with_float = 'float32'
        self.expected_out_dtype_with_complex = 'complex64'

    def generate_test_value(self):
        self.value = paddle.rand([2, 3, 4]).astype(self.dtype)

    def run_api(self):
        prog = paddle.static.Program()
        with paddle.static.program_guard(prog):
            self.generate_test_value()
            out_int = self.value % 1
            out_float = self.value % 1.0
            # mod API not support complex
            res = self.exe.run(prog, fetch_list=[out_int, out_float])
        return res

    def test_dtype_is_expected(self):
        res_int, res_float = self.run_api()
        self.assertEqual(
            res_int.dtype.__str__(), self.expected_out_dtype_with_int
        )
        self.assertEqual(
            res_float.dtype.__str__(), self.expected_out_dtype_with_float
        )
        # self.assertEqual(res_complex.dtype.__str__(), self.expected_out_dtype_with_complex)


# # mod API support float32, float64, int32, int64
create_test_case(TestTensorModScalar, 'int32', 'int32', 'float32', 'complex64')
create_test_case(TestTensorModScalar, 'int64', 'int64', 'float32', 'complex64')
create_test_case(
    TestTensorModScalar, 'float64', 'float64', 'float64', 'complex128'
)


class Test0DTensor(unittest.TestCase):
    def test_0d_add_0d(self):
        paddle.enable_static()
        prog = paddle.static.Program()
        exe = paddle.static.Executor()
        with paddle.static.program_guard(prog):
            a = paddle.ones([], dtype='complex128')
            b = paddle.ones([], dtype='complex64')
            out = a + b
            res = exe.run(prog, fetch_list=[out])
        return res


if __name__ == '__main__':
    unittest.main()
