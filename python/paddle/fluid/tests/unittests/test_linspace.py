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

<<<<<<< HEAD
import unittest

import numpy as np
from op_test import OpTest

import paddle
import paddle.fluid as fluid
from paddle.fluid import Program, core, program_guard


class TestLinspaceOpCommonCase(OpTest):
=======
from __future__ import print_function

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid import core
from paddle.fluid.framework import _test_eager_guard


class TestLinspaceOpCommonCase(OpTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "linspace"
        self.python_api = paddle.linspace
        dtype = 'float32'
        self.inputs = {
            'Start': np.array([0]).astype(dtype),
            'Stop': np.array([10]).astype(dtype),
<<<<<<< HEAD
            'Num': np.array([11]).astype('int32'),
=======
            'Num': np.array([11]).astype('int32')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        self.attrs = {'dtype': int(core.VarDesc.VarType.FP32)}

        self.outputs = {'Out': np.arange(0, 11).astype(dtype)}

    def test_check_output(self):
        self.check_output(check_eager=True)


class TestLinspaceOpReverseCase(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "linspace"
        self.python_api = paddle.linspace
        dtype = 'float32'
        self.inputs = {
            'Start': np.array([10]).astype(dtype),
            'Stop': np.array([0]).astype(dtype),
<<<<<<< HEAD
            'Num': np.array([11]).astype('int32'),
=======
            'Num': np.array([11]).astype('int32')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        self.attrs = {'dtype': int(core.VarDesc.VarType.FP32)}

        self.outputs = {'Out': np.arange(10, -1, -1).astype(dtype)}

    def test_check_output(self):
        self.check_output(check_eager=True)


class TestLinspaceOpNumOneCase(OpTest):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "linspace"
        self.python_api = paddle.linspace
        dtype = 'float32'
        self.inputs = {
            'Start': np.array([10]).astype(dtype),
            'Stop': np.array([0]).astype(dtype),
<<<<<<< HEAD
            'Num': np.array([1]).astype('int32'),
=======
            'Num': np.array([1]).astype('int32')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }
        self.attrs = {'dtype': int(core.VarDesc.VarType.FP32)}

        self.outputs = {'Out': np.array(10, dtype=dtype)}

    def test_check_output(self):
        self.check_output(check_eager=True)


class TestLinspaceAPI(unittest.TestCase):
<<<<<<< HEAD
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_variable_input1(self):
        start = paddle.full(shape=[1], fill_value=0, dtype='float32')
        stop = paddle.full(shape=[1], fill_value=10, dtype='float32')
        num = paddle.full(shape=[1], fill_value=5, dtype='int32')
        out = paddle.linspace(start, stop, num, dtype='float32')
        exe = fluid.Executor(place=fluid.CPUPlace())
        res = exe.run(fluid.default_main_program(), fetch_list=[out])
        np_res = np.linspace(0, 10, 5, dtype='float32')
        self.assertEqual((res == np_res).all(), True)

    def test_variable_input2(self):
        paddle.disable_static()
        start = paddle.full(shape=[1], fill_value=0, dtype='float32')
        stop = paddle.full(shape=[1], fill_value=10, dtype='float32')
        num = paddle.full(shape=[1], fill_value=5, dtype='int32')
        out = paddle.linspace(start, stop, num, dtype='float32')
        np_res = np.linspace(0, 10, 5, dtype='float32')
        self.assertEqual((out.numpy() == np_res).all(), True)
        paddle.enable_static()

    def test_dtype(self):
        out_1 = paddle.linspace(0, 10, 5, dtype='float32')
        out_2 = paddle.linspace(0, 10, 5, dtype=np.float32)
        out_3 = paddle.linspace(0, 10, 5, dtype=core.VarDesc.VarType.FP32)
        exe = fluid.Executor(place=fluid.CPUPlace())
<<<<<<< HEAD
        res_1, res_2, res_3 = exe.run(
            fluid.default_main_program(), fetch_list=[out_1, out_2, out_3]
        )
=======
        res_1, res_2, res_3 = exe.run(fluid.default_main_program(),
                                      fetch_list=[out_1, out_2, out_3])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        assert np.array_equal(res_1, res_2)

    def test_name(self):
        with paddle.static.program_guard(paddle.static.Program()):
<<<<<<< HEAD
            out = paddle.linspace(
                0, 10, 5, dtype='float32', name='linspace_res'
            )
=======
            out = paddle.linspace(0,
                                  10,
                                  5,
                                  dtype='float32',
                                  name='linspace_res')
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            assert 'linspace_res' in out.name

    def test_imperative(self):
        paddle.disable_static()
        out1 = paddle.linspace(0, 10, 5, dtype='float32')
        np_out1 = np.linspace(0, 10, 5, dtype='float32')
        out2 = paddle.linspace(0, 10, 5, dtype='int32')
        np_out2 = np.linspace(0, 10, 5, dtype='int32')
        out3 = paddle.linspace(0, 10, 200, dtype='int32')
        np_out3 = np.linspace(0, 10, 200, dtype='int32')
        paddle.enable_static()
        self.assertEqual((out1.numpy() == np_out1).all(), True)
        self.assertEqual((out2.numpy() == np_out2).all(), True)
        self.assertEqual((out3.numpy() == np_out3).all(), True)

<<<<<<< HEAD

class TestLinspaceOpError(unittest.TestCase):
=======
    def test_api_eager_dygraph(self):
        with _test_eager_guard():
            self.test_variable_input2()
            self.test_imperative()


class TestLinspaceOpError(unittest.TestCase):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def test_errors(self):
        with program_guard(Program(), Program()):

            def test_dtype():
<<<<<<< HEAD
                paddle.linspace(0, 10, 1, dtype="int8")
=======
                fluid.layers.linspace(0, 10, 1, dtype="int8")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertRaises(TypeError, test_dtype)

            def test_dtype1():
<<<<<<< HEAD
                paddle.linspace(0, 10, 1.33, dtype="int32")
=======
                fluid.layers.linspace(0, 10, 1.33, dtype="int32")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertRaises(TypeError, test_dtype1)

            def test_start_type():
<<<<<<< HEAD
                paddle.linspace([0], 10, 1, dtype="float32")
=======
                fluid.layers.linspace([0], 10, 1, dtype="float32")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertRaises(TypeError, test_start_type)

            def test_end_type():
<<<<<<< HEAD
                paddle.linspace(0, [10], 1, dtype="float32")
=======
                fluid.layers.linspace(0, [10], 1, dtype="float32")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertRaises(TypeError, test_end_type)

            def test_step_dtype():
<<<<<<< HEAD
                paddle.linspace(0, 10, [0], dtype="float32")
=======
                fluid.layers.linspace(0, 10, [0], dtype="float32")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertRaises(TypeError, test_step_dtype)

            def test_start_dtype():
                start = fluid.data(shape=[1], dtype="float64", name="start")
<<<<<<< HEAD
                paddle.linspace(start, 10, 1, dtype="float32")
=======
                fluid.layers.linspace(start, 10, 1, dtype="float32")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertRaises(ValueError, test_start_dtype)

            def test_end_dtype():
                end = fluid.data(shape=[1], dtype="float64", name="end")
<<<<<<< HEAD
                paddle.linspace(0, end, 1, dtype="float32")
=======
                fluid.layers.linspace(0, end, 1, dtype="float32")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertRaises(ValueError, test_end_dtype)

            def test_num_dtype():
                num = fluid.data(shape=[1], dtype="int32", name="step")
<<<<<<< HEAD
                paddle.linspace(0, 10, num, dtype="float32")
=======
                fluid.layers.linspace(0, 10, num, dtype="float32")
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

            self.assertRaises(TypeError, test_step_dtype)


if __name__ == "__main__":
    unittest.main()
