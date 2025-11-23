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

import numpy
import numpy as np
import op_test
from utils import dygraph_guard

import paddle
from paddle.base import core


class TestEqualComplex64Api(op_test.OpTest):
    def setUp(self):
        self.op_type = 'equal'
        self.typename = ("float32", "complex64")
        self.dtype = "complex64"
        self.python_api = paddle.equal
        x_real = numpy.random.uniform((6, 5, 4, 3)).astype(self.typename[0])
        x_imag = numpy.random.uniform((6, 5, 4, 3)).astype(self.typename[0])
        y_real = numpy.random.uniform((6, 5, 4, 3)).astype(self.typename[0])
        y_imag = numpy.random.uniform((6, 5, 4, 3)).astype(self.typename[0])
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': self.inputs['X'] == self.inputs['Y']}

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestEqualComplex64InfCase(TestEqualComplex64Api):
    def setUp(self):
        super().setUp()
        x_real = np.array([1, np.inf, -np.inf]).astype(self.typename[0])
        x_imag = np.array([1, -1, 1]).astype(self.typename[0])
        y_real = np.array([1, np.inf, -np.inf]).astype(self.typename[0])
        y_imag = np.array([1, 1, -1]).astype(self.typename[0])
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': self.inputs['X'] == self.inputs['Y']}


class TestEqualComplex64NanCase(TestEqualComplex64Api):
    def setUp(self):
        super().setUp()
        x_real = np.array([1, np.nan, -np.nan]).astype(self.typename[0])
        x_imag = np.array([1, -1, 1]).astype(self.typename[0])
        y_real = np.array([1, np.nan, -np.nan]).astype(self.typename[0])
        y_imag = np.array([1, 1, -1]).astype(self.typename[0])
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': self.inputs['X'] == self.inputs['Y']}


class TestEqualComplex128Api(op_test.OpTest):
    def setUp(self):
        self.op_type = 'equal'
        self.typename = ("float64", "complex128")
        self.dtype = "complex128"
        self.python_api = paddle.equal
        x_real = numpy.random.uniform((6, 5, 4, 3)).astype(self.typename[0])
        x_imag = numpy.random.uniform((6, 5, 4, 3)).astype(self.typename[0])
        y_real = numpy.random.uniform((6, 5, 4, 3)).astype(self.typename[0])
        y_imag = numpy.random.uniform((6, 5, 4, 3)).astype(self.typename[0])
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': self.inputs['X'] == self.inputs['Y']}

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestEqualComplex128InfCase(TestEqualComplex128Api):
    def setUp(self):
        super().setUp()
        x_real = np.array([1, np.inf, -np.inf]).astype(self.typename[0])
        x_imag = np.array([1, -1, 1]).astype(self.typename[0])
        y_real = np.array([1, np.inf, -np.inf]).astype(self.typename[0])
        y_imag = np.array([1, 1, -1]).astype(self.typename[0])
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': self.inputs['X'] == self.inputs['Y']}


class TestEqualComplex128NanCase(TestEqualComplex128Api):
    def setUp(self):
        super().setUp()
        x_real = np.array([1, np.nan, -np.nan]).astype(self.typename[0])
        x_imag = np.array([1, -1, 1]).astype(self.typename[0])
        y_real = np.array([1, np.nan, -np.nan]).astype(self.typename[0])
        y_imag = np.array([1, 1, -1]).astype(self.typename[0])
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': self.inputs['X'] == self.inputs['Y']}


class TestNotEqualComplex64Api(op_test.OpTest):
    def setUp(self):
        self.op_type = 'not_equal'
        self.typename = ("float32", "complex64")
        self.dtype = "complex64"
        self.python_api = paddle.not_equal
        x_real = numpy.random.uniform((6, 5, 4, 3)).astype(self.typename[0])
        x_imag = numpy.random.uniform((6, 5, 4, 3)).astype(self.typename[0])
        y_real = numpy.random.uniform((6, 5, 4, 3)).astype(self.typename[0])
        y_imag = numpy.random.uniform((6, 5, 4, 3)).astype(self.typename[0])
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': self.inputs['X'] != self.inputs['Y']}

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestNotEqualComplex64InfCase(TestNotEqualComplex64Api):
    def setUp(self):
        super().setUp()
        x_real = np.array([1, np.inf, -np.inf]).astype(self.typename[0])
        x_imag = np.array([1, -1, 1]).astype(self.typename[0])
        y_real = np.array([1, np.inf, -np.inf]).astype(self.typename[0])
        y_imag = np.array([1, 1, -1]).astype(self.typename[0])
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': self.inputs['X'] != self.inputs['Y']}


class TestNotEqualComplex64NanCase(TestNotEqualComplex64Api):
    def setUp(self):
        super().setUp()
        x_real = np.array([1, np.nan, -np.nan]).astype(self.typename[0])
        x_imag = np.array([1, -1, 1]).astype(self.typename[0])
        y_real = np.array([1, np.nan, -np.nan]).astype(self.typename[0])
        y_imag = np.array([1, 1, -1]).astype(self.typename[0])
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': self.inputs['X'] != self.inputs['Y']}


class TestNotEqualComplex128Api(op_test.OpTest):
    def setUp(self):
        self.op_type = 'not_equal'
        self.typename = ("float64", "complex128")
        self.dtype = "complex128"
        self.python_api = paddle.not_equal
        x_real = numpy.random.uniform((6, 5, 4, 3)).astype(self.typename[0])
        x_imag = numpy.random.uniform((6, 5, 4, 3)).astype(self.typename[0])
        y_real = numpy.random.uniform((6, 5, 4, 3)).astype(self.typename[0])
        y_imag = numpy.random.uniform((6, 5, 4, 3)).astype(self.typename[0])
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': self.inputs['X'] != self.inputs['Y']}

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestNotEqualComplex128InfCase(TestNotEqualComplex128Api):
    def setUp(self):
        super().setUp()
        x_real = np.array([1, np.inf, -np.inf]).astype(self.typename[0])
        x_imag = np.array([1, -1, 1]).astype(self.typename[0])
        y_real = np.array([1, np.inf, -np.inf]).astype(self.typename[0])
        y_imag = np.array([1, 1, -1]).astype(self.typename[0])
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': self.inputs['X'] != self.inputs['Y']}


class TestNotEqualComplex128NanCase(TestNotEqualComplex128Api):
    def setUp(self):
        super().setUp()
        x_real = np.array([1, np.nan, -np.nan]).astype(self.typename[0])
        x_imag = np.array([1, -1, 1]).astype(self.typename[0])
        y_real = np.array([1, np.nan, -np.nan]).astype(self.typename[0])
        y_imag = np.array([1, 1, -1]).astype(self.typename[0])
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': self.inputs['X'] != self.inputs['Y']}


@unittest.skipIf(
    core.is_compiled_with_xpu(), "core is compiled with XPU, not support..."
)
class TestEqualSpecialCase(unittest.TestCase):
    def test_api_complex64(self):
        with dygraph_guard():
            a_np = np.array(1 + 1j, dtype="complex64")
            a = paddle.to_tensor(1 + 1j, dtype="complex64")
            b = complex(1, 1)
            c_np = a_np == b
            c = a.equal(b)
            np.testing.assert_allclose(c.numpy(), c_np)

    def test_api_complex128(self):
        with dygraph_guard():
            a_np = np.array(1 + 1j, dtype="complex128")
            a = paddle.to_tensor(1 + 1j, dtype="complex128")
            b = complex(1, 1)
            c_np = a_np == b
            c = a.equal(b)
            np.testing.assert_allclose(c.numpy(), c_np)


class TestLessThanComplex64Api(op_test.OpTest):
    def setUp(self):
        self.op_type = 'less_than'
        self.real_dtype = "float32"
        self.dtype = "complex64"
        self.python_api = paddle.less_than
        x_real = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        x_imag = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        y_real = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        y_imag = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        out = np.zeros_like(x, dtype=bool)
        for i in np.ndindex(x.shape):
            if x_real[i] < y_real[i]:
                out[i] = True
            elif x_real[i] == y_real[i] and x_imag[i] < y_imag[i]:
                out[i] = True
            else:
                out[i] = False
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestLessThanComplex64InfCase(TestLessThanComplex64Api):
    def setUp(self):
        super().setUp()
        x_real = np.array([1, np.inf, -np.inf, 0, 0]).astype(self.real_dtype)
        x_imag = np.array([1, -1, 1, np.inf, -np.inf]).astype(self.real_dtype)
        y_real = np.array([2, np.inf, -np.inf, 0, 0]).astype(self.real_dtype)
        y_imag = np.array([1, 1, -1, np.inf, np.inf]).astype(self.real_dtype)
        x = np.array([complex(r, i) for r, i in zip(x_real, x_imag)])
        y = np.array([complex(r, i) for r, i in zip(y_real, y_imag)])
        self.inputs = {'X': x, 'Y': y}
        out = np.zeros_like(x, dtype=bool)
        for i in np.ndindex(x.shape):
            if x_real[i] < y_real[i]:
                out[i] = True
            elif x_real[i] == y_real[i] and x_imag[i] < y_imag[i]:
                out[i] = True
            else:
                out[i] = False
        self.outputs = {'Out': out}


class TestLessThanComplex64NanCase(TestLessThanComplex64Api):
    def setUp(self):
        super().setUp()
        x_real = np.array([1, np.nan, -np.nan, 0, 0]).astype(self.real_dtype)
        x_imag = np.array([1, -1, 1, np.nan, -np.nan]).astype(self.real_dtype)
        y_real = np.array([2, np.nan, -np.nan, 0, 0]).astype(self.real_dtype)
        y_imag = np.array([1, 1, -1, np.nan, np.nan]).astype(self.real_dtype)
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        # NaN comparisons always return False
        self.outputs = {'Out': x < y}


class TestLessThanComplex128Api(op_test.OpTest):
    def setUp(self):
        self.op_type = 'less_than'
        self.real_dtype = "float64"
        self.dtype = "complex128"
        self.python_api = paddle.less_than
        x_real = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        x_imag = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        y_real = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        y_imag = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        out = np.zeros_like(x, dtype=bool)
        for i in np.ndindex(x.shape):
            if x_real[i] < y_real[i]:
                out[i] = True
            elif x_real[i] == y_real[i] and x_imag[i] < y_imag[i]:
                out[i] = True
            else:
                out[i] = False
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestLessThanComplex128InfCase(TestLessThanComplex128Api):
    def setUp(self):
        super().setUp()
        x_real = np.array([1, np.inf, -np.inf, 0, 0]).astype(self.real_dtype)
        x_imag = np.array([1, -1, 1, np.inf, -np.inf]).astype(self.real_dtype)
        y_real = np.array([2, np.inf, -np.inf, 0, 0]).astype(self.real_dtype)
        y_imag = np.array([1, 1, -1, np.inf, np.inf]).astype(self.real_dtype)
        x = np.array([complex(r, i) for r, i in zip(x_real, x_imag)])
        y = np.array([complex(r, i) for r, i in zip(y_real, y_imag)])
        self.inputs = {'X': x, 'Y': y}
        out = np.zeros_like(x, dtype=bool)
        for i in np.ndindex(x.shape):
            if x_real[i] < y_real[i]:
                out[i] = True
            elif x_real[i] == y_real[i] and x_imag[i] < y_imag[i]:
                out[i] = True
            else:
                out[i] = False
        self.outputs = {'Out': out}


class TestLessThanComplex128NanCase(TestLessThanComplex128Api):
    def setUp(self):
        super().setUp()
        x_real = np.array([1, np.nan, -np.nan, 0, 0]).astype(self.real_dtype)
        x_imag = np.array([1, -1, 1, np.nan, -np.nan]).astype(self.real_dtype)
        y_real = np.array([2, np.nan, -np.nan, 0, 0]).astype(self.real_dtype)
        y_imag = np.array([1, 1, -1, np.nan, np.nan]).astype(self.real_dtype)
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': x < y}


class TestLessEqualComplex64Api(op_test.OpTest):
    def setUp(self):
        self.op_type = 'less_equal'
        self.real_dtype = "float32"
        self.dtype = "complex64"
        self.python_api = paddle.less_equal
        x_real = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        x_imag = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        y_real = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        y_imag = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        out = np.zeros_like(x, dtype=bool)
        for i in np.ndindex(x.shape):
            if x_real[i] < y_real[i]:
                out[i] = True
            elif x_real[i] == y_real[i] and x_imag[i] <= y_imag[i]:
                out[i] = True
            else:
                out[i] = False
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestLessEqualComplex64InfCase(TestLessEqualComplex64Api):
    def setUp(self):
        super().setUp()
        x_real = np.array([1, np.inf, -np.inf, 0, 0, 1]).astype(self.real_dtype)
        x_imag = np.array([1, -1, 1, np.inf, -np.inf, 1]).astype(
            self.real_dtype
        )
        y_real = np.array([2, np.inf, -np.inf, 0, 0, 1]).astype(self.real_dtype)
        y_imag = np.array([1, 1, -1, np.inf, np.inf, 1]).astype(self.real_dtype)
        x = np.array([complex(r, i) for r, i in zip(x_real, x_imag)])
        y = np.array([complex(r, i) for r, i in zip(y_real, y_imag)])
        self.inputs = {'X': x, 'Y': y}
        out = np.zeros_like(x, dtype=bool)
        for i in np.ndindex(x.shape):
            if x_real[i] < y_real[i]:
                out[i] = True
            elif x_real[i] == y_real[i] and x_imag[i] <= y_imag[i]:
                out[i] = True
            else:
                out[i] = False
        self.outputs = {'Out': out}


class TestLessEqualComplex64NanCase(TestLessEqualComplex64Api):
    def setUp(self):
        super().setUp()
        x_real = np.array([1, np.nan, -np.nan, 0, 0]).astype(self.real_dtype)
        x_imag = np.array([1, -1, 1, np.nan, -np.nan]).astype(self.real_dtype)
        y_real = np.array([2, np.nan, -np.nan, 0, 0]).astype(self.real_dtype)
        y_imag = np.array([1, 1, -1, np.nan, np.nan]).astype(self.real_dtype)
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': x <= y}


class TestLessEqualComplex128Api(op_test.OpTest):
    def setUp(self):
        self.op_type = 'less_equal'
        self.real_dtype = "float64"
        self.dtype = "complex128"
        self.python_api = paddle.less_equal
        x_real = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        x_imag = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        y_real = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        y_imag = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        out = np.zeros_like(x, dtype=bool)
        for i in np.ndindex(x.shape):
            if x_real[i] < y_real[i]:
                out[i] = True
            elif x_real[i] == y_real[i] and x_imag[i] <= y_imag[i]:
                out[i] = True
            else:
                out[i] = False
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestLessEqualComplex128InfCase(TestLessEqualComplex128Api):
    def setUp(self):
        super().setUp()
        x_real = np.array([1, np.inf, -np.inf, 0, 0, 1]).astype(self.real_dtype)
        x_imag = np.array([1, -1, 1, np.inf, -np.inf, 1]).astype(
            self.real_dtype
        )
        y_real = np.array([2, np.inf, -np.inf, 0, 0, 1]).astype(self.real_dtype)
        y_imag = np.array([1, 1, -1, np.inf, np.inf, 1]).astype(self.real_dtype)
        x = np.array([complex(r, i) for r, i in zip(x_real, x_imag)])
        y = np.array([complex(r, i) for r, i in zip(y_real, y_imag)])
        self.inputs = {'X': x, 'Y': y}
        out = np.zeros_like(x, dtype=bool)
        for i in np.ndindex(x.shape):
            if x_real[i] < y_real[i]:
                out[i] = True
            elif x_real[i] == y_real[i] and x_imag[i] <= y_imag[i]:
                out[i] = True
            else:
                out[i] = False
        self.outputs = {'Out': out}


class TestLessEqualComplex128NanCase(TestLessEqualComplex128Api):
    def setUp(self):
        super().setUp()
        x_real = np.array([1, np.nan, -np.nan, 0, 0]).astype(self.real_dtype)
        x_imag = np.array([1, -1, 1, np.nan, -np.nan]).astype(self.real_dtype)
        y_real = np.array([2, np.nan, -np.nan, 0, 0]).astype(self.real_dtype)
        y_imag = np.array([1, 1, -1, np.nan, np.nan]).astype(self.real_dtype)
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': x <= y}


class TestGreaterThanComplex64Api(op_test.OpTest):
    def setUp(self):
        self.op_type = 'greater_than'
        self.real_dtype = "float32"
        self.dtype = "complex64"
        self.python_api = paddle.greater_than
        x_real = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        x_imag = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        y_real = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        y_imag = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        out = np.zeros_like(x, dtype=bool)
        for i in np.ndindex(x.shape):
            if x_real[i] > y_real[i]:
                out[i] = True
            elif x_real[i] == y_real[i] and x_imag[i] > y_imag[i]:
                out[i] = True
            else:
                out[i] = False
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestGreaterThanComplex64InfCase(TestGreaterThanComplex64Api):
    def setUp(self):
        super().setUp()
        x_real = np.array([2, np.inf, -np.inf, 0, 0]).astype(self.real_dtype)
        x_imag = np.array([1, 1, -1, np.inf, np.inf]).astype(self.real_dtype)
        y_real = np.array([1, np.inf, -np.inf, 0, 0]).astype(self.real_dtype)
        y_imag = np.array([1, -1, 1, np.inf, -np.inf]).astype(self.real_dtype)
        x = np.array([complex(r, i) for r, i in zip(x_real, x_imag)])
        y = np.array([complex(r, i) for r, i in zip(y_real, y_imag)])

        self.inputs = {'X': x, 'Y': y}
        out = np.zeros_like(x, dtype=bool)
        for i in np.ndindex(x.shape):
            if x_real[i] > y_real[i]:
                out[i] = True
            elif x_real[i] == y_real[i] and x_imag[i] > y_imag[i]:
                out[i] = True
            else:
                out[i] = False
        self.outputs = {'Out': out}


class TestGreaterThanComplex64NanCase(TestGreaterThanComplex64Api):
    def setUp(self):
        super().setUp()
        x_real = np.array([2, np.nan, -np.nan, 0, 0]).astype(self.real_dtype)
        x_imag = np.array([1, 1, -1, np.nan, np.nan]).astype(self.real_dtype)
        y_real = np.array([1, np.nan, -np.nan, 0, 0]).astype(self.real_dtype)
        y_imag = np.array([1, -1, 1, np.nan, -np.nan]).astype(self.real_dtype)
        x = np.array([complex(r, i) for r, i in zip(x_real, x_imag)])
        y = np.array([complex(r, i) for r, i in zip(y_real, y_imag)])
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': x > y}


class TestGreaterThanComplex128Api(op_test.OpTest):
    def setUp(self):
        self.op_type = 'greater_than'
        self.real_dtype = "float64"
        self.dtype = "complex128"
        self.python_api = paddle.greater_than
        x_real = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        x_imag = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        y_real = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        y_imag = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        out = np.zeros_like(x, dtype=bool)
        for i in np.ndindex(x.shape):
            if x_real[i] > y_real[i]:
                out[i] = True
            elif x_real[i] == y_real[i] and x_imag[i] > y_imag[i]:
                out[i] = True
            else:
                out[i] = False
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestGreaterThanComplex128InfCase(TestGreaterThanComplex128Api):
    def setUp(self):
        super().setUp()
        x_real = np.array([2, np.inf, -np.inf, 0, 0]).astype(self.real_dtype)
        x_imag = np.array([1, 1, -1, np.inf, np.inf]).astype(self.real_dtype)
        y_real = np.array([1, np.inf, -np.inf, 0, 0]).astype(self.real_dtype)
        y_imag = np.array([1, -1, 1, np.inf, -np.inf]).astype(self.real_dtype)
        x = np.array([complex(r, i) for r, i in zip(x_real, x_imag)])
        y = np.array([complex(r, i) for r, i in zip(y_real, y_imag)])
        self.inputs = {'X': x, 'Y': y}
        out = np.zeros_like(x, dtype=bool)
        for i in np.ndindex(x.shape):
            if x_real[i] > y_real[i]:
                out[i] = True
            elif x_real[i] == y_real[i] and x_imag[i] > y_imag[i]:
                out[i] = True
            else:
                out[i] = False
        self.outputs = {'Out': out}


class TestGreaterThanComplex128NanCase(TestGreaterThanComplex128Api):
    def setUp(self):
        super().setUp()
        x_real = np.array([2, np.nan, -np.nan, 0, 0]).astype(self.real_dtype)
        x_imag = np.array([1, 1, -1, np.nan, np.nan]).astype(self.real_dtype)
        y_real = np.array([1, np.nan, -np.nan, 0, 0]).astype(self.real_dtype)
        y_imag = np.array([1, -1, 1, np.nan, -np.nan]).astype(self.real_dtype)
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': x > y}


class TestGreaterEqualComplex64Api(op_test.OpTest):
    def setUp(self):
        self.op_type = 'greater_equal'
        self.real_dtype = "float32"
        self.dtype = "complex64"
        self.python_api = paddle.greater_equal
        x_real = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        x_imag = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        y_real = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        y_imag = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        out = np.zeros_like(x, dtype=bool)
        for i in np.ndindex(x.shape):
            if x_real[i] > y_real[i]:
                out[i] = True
            elif x_real[i] == y_real[i] and x_imag[i] >= y_imag[i]:
                out[i] = True
            else:
                out[i] = False
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestGreaterEqualComplex64InfCase(TestGreaterEqualComplex64Api):
    def setUp(self):
        super().setUp()
        x_real = np.array([2, np.inf, -np.inf, 0, 0, 1]).astype(self.real_dtype)
        x_imag = np.array([1, 1, -1, np.inf, np.inf, 1]).astype(self.real_dtype)
        y_real = np.array([1, np.inf, -np.inf, 0, 0, 1]).astype(self.real_dtype)
        y_imag = np.array([1, -1, 1, np.inf, -np.inf, 1]).astype(
            self.real_dtype
        )
        x = np.array([complex(r, i) for r, i in zip(x_real, x_imag)])
        y = np.array([complex(r, i) for r, i in zip(y_real, y_imag)])
        self.inputs = {'X': x, 'Y': y}
        out = np.zeros_like(x, dtype=bool)
        for i in np.ndindex(x.shape):
            if x_real[i] > y_real[i]:
                out[i] = True
            elif x_real[i] == y_real[i] and x_imag[i] >= y_imag[i]:
                out[i] = True
            else:
                out[i] = False
        self.outputs = {'Out': out}


class TestGreaterEqualComplex64NanCase(TestGreaterEqualComplex64Api):
    def setUp(self):
        super().setUp()
        x_real = np.array([2, np.nan, -np.nan, 0, 0]).astype(self.real_dtype)
        x_imag = np.array([1, 1, -1, np.nan, np.nan]).astype(self.real_dtype)
        y_real = np.array([1, np.nan, -np.nan, 0, 0]).astype(self.real_dtype)
        y_imag = np.array([1, -1, 1, np.nan, -np.nan]).astype(self.real_dtype)
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': x >= y}


class TestGreaterEqualComplex128Api(op_test.OpTest):
    def setUp(self):
        self.op_type = 'greater_equal'
        self.real_dtype = "float64"
        self.dtype = "complex128"
        self.python_api = paddle.greater_equal
        x_real = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        x_imag = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        y_real = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        y_imag = numpy.random.uniform(-10, 10, (6, 5, 4, 3)).astype(
            self.real_dtype
        )
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        out = np.zeros_like(x, dtype=bool)
        for i in np.ndindex(x.shape):
            if x_real[i] > y_real[i]:
                out[i] = True
            elif x_real[i] == y_real[i] and x_imag[i] >= y_imag[i]:
                out[i] = True
            else:
                out[i] = False
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(check_pir=True)


class TestGreaterEqualComplex128InfCase(TestGreaterEqualComplex128Api):
    def setUp(self):
        super().setUp()
        x_real = np.array([2, np.inf, -np.inf, 0, 0, 1]).astype(self.real_dtype)
        x_imag = np.array([1, 1, -1, np.inf, np.inf, 1]).astype(self.real_dtype)
        y_real = np.array([1, np.inf, -np.inf, 0, 0, 1]).astype(self.real_dtype)
        y_imag = np.array([1, -1, 1, np.inf, -np.inf, 1]).astype(
            self.real_dtype
        )
        x = np.array([complex(r, i) for r, i in zip(x_real, x_imag)])
        y = np.array([complex(r, i) for r, i in zip(y_real, y_imag)])
        self.inputs = {'X': x, 'Y': y}
        out = np.zeros_like(x, dtype=bool)
        for i in np.ndindex(x.shape):
            if x_real[i] > y_real[i]:
                out[i] = True
            elif x_real[i] == y_real[i] and x_imag[i] >= y_imag[i]:
                out[i] = True
            else:
                out[i] = False
        self.outputs = {'Out': out}


class TestGreaterEqualComplex128NanCase(TestGreaterEqualComplex128Api):
    def setUp(self):
        super().setUp()
        x_real = np.array([2, np.nan, -np.nan, 0, 0]).astype(self.real_dtype)
        x_imag = np.array([1, 1, -1, np.nan, np.nan]).astype(self.real_dtype)
        y_real = np.array([1, np.nan, -np.nan, 0, 0]).astype(self.real_dtype)
        y_imag = np.array([1, -1, 1, np.nan, -np.nan]).astype(self.real_dtype)
        x = x_real + 1j * x_imag
        y = y_real + 1j * y_imag
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': x >= y}


if __name__ == '__main__':
    unittest.main()
