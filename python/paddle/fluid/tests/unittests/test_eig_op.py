#  Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from op_test import OpTest, skip_check_grad_ci
import unittest
from paddle.fluid.op import Operator
from paddle.fluid import compiler, Program, program_guard


@skip_check_grad_ci(reason="temp")
class TestEigOp(OpTest):
    def setUp(self):
        self.op_type = "eig"
        self.__class__.op_type = self.op_type

        ipt = np.random.random((3, 3)) + np.random.random((3, 3)) * 1j
        #ipt = np.random.random((3,3)).astype('float32') #float64
        self.inputs = {'X': ipt}
        w, v = np.linalg.eig(ipt)
        self.outputs = {'OutValues': w, 'OutVectors': v}

    def test_check_output(self):
        # numpy 输出实数，eig输出复数
        self.check_output_with_place(place=core.CPUPlace(), check_dygraph=True)

    def test_grad(self):
        pass
        #self.check_grad(["X"], "OutValues", check_dygraph=True)

    # if __name__ == "__main__":
    #     paddle.enable_static()
    #     unittest.main()


'''
@skip_check_grad_ci(reason="temp") # no grad test
class TestEigOp(OpTest):
    def setUp(self):
        self.op_type = "eig"
        self.__class__.op_type = self.op_type
        self.init_input()
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(self.x)}
        self.outputs = {
            'OutValues': self.out[0],
            'OutVectors': self.out[1]
        }

    def init_input(self):
        self.shape = (3, 3)
        self.dtype = 'float32'
        self.x = np.random.random(self.shape).astype(self.dtype)
        self.out = np.linalg.eig(self.x)

    def compare_results(self, expect, actual, rtol, atol, place):
        self.assertTrue(np.allclose(
            np.abs(expect[0]), paddle.abs(actual[0]), rtol, atol, place),
            "Eigen values has diff at " + str(place) +
            "\nExpect " + str(expect[0]) + "\n" + "But Got" +
            str(actual[0]) + " in class " + self.__class__.__name__)
        
        self.assertTrue(np.allclose(
            np.abs(expect[1]), paddle.abs(actual[1]), rtol, atol, place),
            "Eigen vectors has diff at " + str(place) +
            "\nExpect " + str(expect[1]) + "\n" + "But Got" +
            str(actual[1]) + " in class " + self.__class__.__name__)

    def test_check_output_with_place(self):
        paddle.disable_static()
        place = fluid.CPUPlace()
        x = paddle.to_tensor(self.x)
        actual = paddle.linalg.eig(paddle.to_tensor(self.x))
        self.compare_results(self.out, actual, 1e-6, 1e-6, place)
        paddle.enable_static()

    def test_grad(self):
        paddle.disable_static()
        #self.check_grad(["X"], "OutValues")
        paddle.enable_static()
'''
'''
class TestEigBatchMarices(TestEigOp):
    def init_input(self):
        self.shape = (3, 3, 3)
        self.dtype = 'float32'
        self.x = np.random.random(self.shape).astype(self.dtype)
        self.out = np.linalg.eig(self.x)


class TestEigStatic(TestEigOp):
    def test_check_output_with_place(self):
        paddle.enable_static()
        place = core.CPUPlace()
        input_np = np.random.random([3,3]).astype('float32')
        expect_val, expect_vec = np.linalg.eig(input_np)
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input = fluid.data(name="input", shape=[3,3], dtype='float32')
            act_val, act_vec = paddle.linalg.eig(input)

            exe = fluid.Executor(place)
            fetch_val, fetch_vec = exe.run(fluid.default_main_program(),
                                            feed={"input": input_np},
                                            fetch_list=[act_val, act_vec])

        self.assertTrue(np.allclose(expect_val, fetch_val, 1e-6, 1e-6), 
                        "The eigen values have diff ")
        self.assertTrue(np.allclose(np.abs(expect_vec), np.abs(fetch_vec), 1e-6, 1e-6), 
                        "The eigen vectors have diff ")

        paddle.disable_static()
'''


class TestEigWrongDimsError(unittest.TestCase):
    def test_error(self):
        paddle.disable_static()
        a = np.random.random((3)).astype('float32')
        x = paddle.to_tensor(a)
        self.assertRaises(IndexError, paddle.linalg.eig, x)
        paddle.enable_static()


class TestEigNotSquareError(unittest.TestCase):
    def test_error(self):
        paddle.disable_static()
        a = np.random.random((1, 2, 3)).astype('float32')
        x = paddle.to_tensor(a)
        self.assertRaises(ValueError, paddle.linalg.eig, x)
        paddle.enable_static()


class TestEigUnsupportedDtypeError(unittest.TestCase):
    def test_error(self):
        paddle.disable_static()
        a = (np.random.random((3, 3)) * 10).astype('int64')
        x = paddle.to_tensor(a)
        self.assertRaises(ValueError, paddle.linalg.eig, x)
        paddle.enable_static()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
