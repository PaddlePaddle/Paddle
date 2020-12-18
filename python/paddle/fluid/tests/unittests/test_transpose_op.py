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

from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard

paddle.enable_static()

class TestTransposeOp(OpTest):
    def setUp(self):
        self.init_op_type()
        self.initTestCase()
        self.inputs = {'X': np.random.random(self.shape).astype("float64")}
        self.attrs = {
            'axis': list(self.axis),
            'use_mkldnn': self.use_mkldnn,
        }
        self.outputs = {
            'XShape': np.random.random(self.shape).astype("float64"),
            'Out': self.inputs['X'].transpose(self.axis)
        }

    def init_op_type(self):
        self.op_type = "transpose2"
        self.use_mkldnn = False

    def test_check_output(self):
        self.check_output(no_check_set=['XShape'])

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')

    def initTestCase(self):
        self.shape = (3, 40)
        self.axis = (1, 0)


class TestCase0(TestTransposeOp):
    def initTestCase(self):
        self.shape = (100, )
        self.axis = (0, )


class TestCase1(TestTransposeOp):
    def initTestCase(self):
        self.shape = (3, 4, 10)
        self.axis = (0, 2, 1)


class TestCase2(TestTransposeOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5)
        self.axis = (0, 2, 3, 1)


class TestCase3(TestTransposeOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6)
        self.axis = (4, 2, 3, 1, 0)


class TestCase4(TestTransposeOp):
    def initTestCase(self):
        self.shape = (2, 3, 4, 5, 6, 1)
        self.axis = (4, 2, 3, 1, 0, 5)


class TestCase5(TestTransposeOp):
    def initTestCase(self):
        self.shape = (2, 16, 96)
        self.axis = (0, 2, 1)


class TestCase6(TestTransposeOp):
    def initTestCase(self):
        self.shape = (2, 10, 12, 16)
        self.axis = (3, 1, 2, 0)


class TestCase7(TestTransposeOp):
    def initTestCase(self):
        self.shape = (2, 10, 2, 16)
        self.axis = (0, 1, 3, 2)


class TestCase8(TestTransposeOp):
    def initTestCase(self):
        self.shape = (2, 3, 2, 3, 2, 4, 3, 3)
        self.axis = (0, 1, 3, 2, 4, 5, 6, 7)


class TestCase9(TestTransposeOp):
    def initTestCase(self):
        self.shape = (2, 3, 2, 3, 2, 4, 3, 3)
        self.axis = (6, 1, 3, 5, 0, 2, 4, 7)


class TestTransposeOpError(unittest.TestCase):
    def test_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            x = fluid.layers.data(name='x', shape=[10, 5, 3], dtype='float64')

            def test_x_Variable_check():
                # the Input(x)'s type must be Variable
                fluid.layers.transpose("not_variable", perm=[1, 0, 2])

            self.assertRaises(TypeError, test_x_Variable_check)

            def test_x_dtype_check():
                # the Input(x)'s dtype must be one of [float16, float32, float64, int32, int64]
                x1 = fluid.layers.data(
                    name='x1', shape=[10, 5, 3], dtype='bool')
                fluid.layers.transpose(x1, perm=[1, 0, 2])

            self.assertRaises(TypeError, test_x_dtype_check)

            def test_perm_list_check():
                # Input(perm)'s type must be list
                fluid.layers.transpose(x, perm="[1, 0, 2]")

            self.assertRaises(TypeError, test_perm_list_check)

            def test_perm_length_and_x_dim_check():
                # Input(perm) is the permutation of dimensions of Input(input)
                # its length should be equal to dimensions of Input(input)
                fluid.layers.transpose(x, perm=[1, 0, 2, 3, 4])

            self.assertRaises(ValueError, test_perm_length_and_x_dim_check)

            def test_each_elem_value_check():
                # Each element in Input(perm) should be less than Input(x)'s dimension
                fluid.layers.transpose(x, perm=[3, 5, 7])

            self.assertRaises(ValueError, test_each_elem_value_check)

class TestTransposeApi(unittest.TestCase):
    def test_static_out(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data(name='x', shape=[2, 3, 4], dtype='float32')
            x_trans1 = paddle.transpose(x, perm=[1, 0, 2])
            x_trans2 = paddle.transpose(x, perm=(2, 1, 0))
            place = paddle.CPUPlace()
            exe = paddle.static.Executor(place)
            x_np = np.random.random([2, 3, 4]).astype("float32")
            result1, result2 = exe.run(feed={"x": x_np}, fetch_list=[x_trans1, x_trans2])
            expected_result1 = np.transpose(x_np, [1, 0, 2])
            expected_result2 = np.transpose(x_np, (2, 1, 0))
            
            np.testing.assert_array_equal(result1, expected_result1)
            np.testing.assert_array_equal(result2, expected_result2)

    def test_dygraph_out(self):
        # This is an old test before 2.0 API so we need to disable static
        # to trigger dygraph
        paddle.disable_static()
        x = paddle.randn([2, 3, 4])
        x_trans1 = paddle.transpose(x, perm=[1, 0, 2])
        x_trans2 = paddle.transpose(x, perm=(2, 1, 0))
        x_np = x.numpy()
        expected_result1 = np.transpose(x_np, [1, 0, 2])
        expected_result2 = np.transpose(x_np, (2, 1, 0))

        np.testing.assert_array_equal(x_trans1.numpy(), expected_result1)
        np.testing.assert_array_equal(x_trans2.numpy(), expected_result2)
        # This is an old test before 2.0 API so we enable static again after
        # dygraph test
        paddle.enable_static()

class TestTAPI(unittest.TestCase):
    def test_out(self):
        with fluid.program_guard(fluid.Program()):
            data = fluid.data(shape=[10], dtype="float64", name="data")
            data_t = paddle.t(data)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            data_np = np.random.random([10]).astype("float64")
            result, = exe.run(feed={"data": data_np}, fetch_list=[data_t])
            expected_result = np.transpose(data_np)
        self.assertEqual((result == expected_result).all(), True)

        with fluid.program_guard(fluid.Program()):
            data = fluid.data(shape=[10, 5], dtype="float64", name="data")
            data_t = paddle.t(data)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            data_np = np.random.random([10, 5]).astype("float64")
            result, = exe.run(feed={"data": data_np}, fetch_list=[data_t])
            expected_result = np.transpose(data_np)
        self.assertEqual((result == expected_result).all(), True)

        with fluid.program_guard(fluid.Program()):
            data = fluid.data(shape=[1, 5], dtype="float64", name="data")
            data_t = paddle.t(data)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            data_np = np.random.random([1, 5]).astype("float64")
            result, = exe.run(feed={"data": data_np}, fetch_list=[data_t])
            expected_result = np.transpose(data_np)
        self.assertEqual((result == expected_result).all(), True)

        with fluid.dygraph.guard():
            np_x = np.random.random([10]).astype("float64")
            data = fluid.dygraph.to_variable(np_x)
            z = paddle.t(data)
            np_z = z.numpy()
            z_expected = np.array(np.transpose(np_x))
        self.assertEqual((np_z == z_expected).all(), True)

        with fluid.dygraph.guard():
            np_x = np.random.random([10, 5]).astype("float64")
            data = fluid.dygraph.to_variable(np_x)
            z = paddle.t(data)
            np_z = z.numpy()
            z_expected = np.array(np.transpose(np_x))
        self.assertEqual((np_z == z_expected).all(), True)

        with fluid.dygraph.guard():
            np_x = np.random.random([1, 5]).astype("float64")
            data = fluid.dygraph.to_variable(np_x)
            z = paddle.t(data)
            np_z = z.numpy()
            z_expected = np.array(np.transpose(np_x))
        self.assertEqual((np_z == z_expected).all(), True)

    def test_errors(self):
        with fluid.program_guard(fluid.Program()):
            x = fluid.data(name='x', shape=[10, 5, 3], dtype='float64')

            def test_x_dimension_check():
                paddle.t(x)

            self.assertRaises(ValueError, test_x_dimension_check)


if __name__ == '__main__':
    unittest.main()
