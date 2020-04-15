# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import Program, program_guard


class BaseTestCase(OpTest):
    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (3, 4, 5)
        self.dtype = 'float32'
        self.axis = 0

    def setUp(self):
        self.initTestCase()
        self.x = (1000 * np.random.random(self.dims)).astype(self.dtype)
        self.inputs = {'X': self.x}
        self.attrs = {'axis': self.axis}
        if self.op_type == "arg_min":
            self.outputs = {'Out': np.argmin(self.x, axis=self.axis)}
        else:
            self.outputs = {'Out': np.argmax(self.x, axis=self.axis)}

    def test_check_output(self):
        self.check_output()


class TestCase0(BaseTestCase):
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, 4, 5)
        self.dtype = 'float32'
        self.axis = 0


class TestCase1(BaseTestCase):
    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (3, 4)
        self.dtype = 'float64'
        self.axis = 1


class TestCase2(BaseTestCase):
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, 4)
        self.dtype = 'int64'
        self.axis = 0


class TestCase2_1(BaseTestCase):
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, 4)
        self.dtype = 'int64'
        self.axis = -1


class TestCase3(BaseTestCase):
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, )
        self.dtype = 'int64'
        self.axis = 0


class TestCase4(BaseTestCase):
    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (1, )
        self.dtype = 'int32'
        self.axis = 0


class TestCase3(BaseTestCase):
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (3, )
        self.axis = 0


class BaseTestComplex1_1(OpTest):
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (4, 5, 6)
        self.dtype = 'int32'
        self.axis = 2

    def setUp(self):
        self.initTestCase()
        self.x = (np.random.random(self.dims)).astype(self.dtype)
        self.inputs = {'X': self.x}
        self.attrs = {'axis': self.axis}
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT32)}
        if self.op_type == "arg_min":
            self.outputs = {
                'Out': np.argmin(
                    self.x, axis=self.axis).asdtype("int32")
            }
        else:
            self.outputs = {
                'Out': np.argmax(
                    self.x, axis=self.axis).asdtype("int32")
            }


class BaseTestComplex1_2(OpTest):
    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (4, 5, 6)
        self.dtype = 'int32'
        self.axis = 2

    def setUp(self):
        self.initTestCase()
        self.x = (np.random.random(self.dims)).astype(self.dtype)
        self.inputs = {'X': self.x}
        self.attrs = {'axis': self.axis}
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT32)}
        if self.op_type == "arg_min":
            self.outputs = {
                'Out': np.argmin(
                    self.x, axis=self.axis).asdtype("int32")
            }
        else:
            self.outputs = {
                'Out': np.argmax(
                    self.x, axis=self.axis).asdtype("int32")
            }


class BaseTestComplex2_1(OpTest):
    def initTestCase(self):
        self.op_type = 'arg_max'
        self.dims = (4, 5, 6)
        self.dtype = 'int32'
        self.axis = 2

    def setUp(self):
        self.initTestCase()
        self.x = (np.random.random(self.dims)).astype(self.dtype)
        self.inputs = {'X': self.x}
        self.attrs = {'axis': self.axis}
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT32)}
        self.attrs = {'keep_dims': True}
        if self.op_type == "arg_min":
            self.outputs = {
                'Out': np.argmin(
                    self.x, axis=self.axis).asdtype("int32").reshape(4, 5, 1)
            }
        else:
            self.outputs = {
                'Out': np.argmax(
                    self.x, axis=self.axis).asdtype("int32").reshape(4, 5, 1)
            }


class BaseTestComplex2_2(OpTest):
    def initTestCase(self):
        self.op_type = 'arg_min'
        self.dims = (4, 5, 6)
        self.dtype = 'int32'
        self.axis = 2

    def setUp(self):
        self.initTestCase()
        self.x = (np.random.random(self.dims)).astype(self.dtype)
        self.inputs = {'X': self.x}
        self.attrs = {'axis': self.axis}
        self.attrs = {'dtype': int(core.VarDesc.VarType.INT32)}
        self.attrs = {'keep_dims': True}
        if self.op_type == "arg_min":
            self.outputs = {
                'Out': np.argmin(
                    self.x, axis=self.axis).asdtype("int32").reshape(4, 5, 1)
            }
        else:
            self.outputs = {
                'Out': np.argmax(
                    self.x, axis=self.axis).asdtype("int32").reshape(4, 5, 1)
            }


class APT_ArgMaxTest(unittest.TestCase):
    def test_output_result(self):
        with fluid.program_guard(fluid.Program()):
            data1 = fluid.data(name="X", shape=[3, 4], dtype="float32")
            data2 = fluid.data(name="Y", shape=[3], dtype="int64")
            out = paddle.argmax(input=data1, out=data2)

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            result = exe.run(
                feed={"X": np.random.rand(3, 4).astype("float32")},
                fetch_list=[data2, out])
            self.assertEqual((result[0] == result[1]).all(), True)

    def test_basic(self):
        with fluid.program_guard(fluid.Program()):
            data = fluid.data(name="X", shape=[3, 4], dtype="float32")
            out = paddle.argmax(input=data)

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            np_input = np.random.rand(3, 4).astype("float32")
            expected_result = np.argmax(np_input, axis=1)

            result, = exe.run(feed={"X": np_input}, fetch_list=[out])
        self.assertEqual((result == expected_result).all(), True)

        with fluid.program_guard(fluid.Program()):
            data = fluid.data(name="X", shape=[3, 4], dtype="float32")
            out = paddle.argmax(input=data, axis=0)

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            np_input = np.random.rand(3, 4).astype("float32")
            expected_result = np.argmax(np_input, axis=0)

            result = exe.run(feed={"X": np_input}, fetch_list=[out])
        self.assertEqual((result == expected_result).all(), True)

        with fluid.program_guard(fluid.Program()):
            data = fluid.data(name="X", shape=[3, 4], dtype="float32")
            out = paddle.argmax(input=data, dtype="int32")

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            np_input = np.random.rand(3, 4).astype("float32")
            expected_result = np.argmax(np_input, axis=1).astype(np.int32)

            result = exe.run(feed={"X": np_input}, fetch_list=[out])
        self.assertEqual((result == expected_result).all(), True)

        with fluid.program_guard(fluid.Program()):
            data1 = fluid.data(name="X", shape=[3, 4], dtype="float32")
            data2 = fluid.data(name="Y", shape=[3], dtype="int64")
            out = paddle.argmax(input=data, out=data2)

            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            result = exe.run(
                feed={"X": np.random.rand(3, 4).astype("float32")},
                fetch_list=[data2, out])
        self.assertEqual((result[0] == result[1]).all(), True)

    def test_name(self):
        with fluid.program_guard(fluid.Program()):
            x = fluid.data(name="x", shape=[100], dtype="float32")
            y_1 = paddle.argmax(x, name='arg_max_res')
            self.assertEqual(('arg_max_res' in y_1.name), True)

    def test_errors(self):
        def test_dtype1():
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                data = fluid.data(name="data", shape=[10], dtype="float32")
                paddle.argmax(data, dtype="float32")

        self.assertRaises(TypeError, test_dtype1)

        def test_dtype2():
            with fluid.program_guard(fluid.Program(), fluid.Program()):
                data = fluid.data(name="data", shape=[10], dtype="float64")
                paddle.argmax(data, dtype="float32")

        self.assertRaises(TypeError, test_dtype2)


class TestArgMinMaxOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):

            def test_argmax_x_type():
                x1 = [1, 2, 3]
                output = fluid.layers.argmax(x=x1)

            self.assertRaises(TypeError, test_argmax_x_type)

            def test_argmin_x_type():
                x2 = [1, 2, 3]
                output = fluid.layers.argmin(x=x2)

            self.assertRaises(TypeError, test_argmin_x_type)


if __name__ == '__main__':
    unittest.main()
