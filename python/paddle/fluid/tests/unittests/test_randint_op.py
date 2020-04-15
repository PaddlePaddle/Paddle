# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
import paddle


def output_hist(out):
    hist, _ = np.histogram(out, range=(-10, 10))
    hist = hist.astype("float32")
    hist /= float(out.size)
    prob = 0.1 * np.ones((10))
    return hist, prob


class TestRandintOp(OpTest):
    def setUp(self):
        self.op_type = "randint"
        self.inputs = {}
        self.init_attrs()
        self.outputs = {"Out": np.zeros((10000, 784)).astype("float32")}

    def init_attrs(self):
        self.attrs = {"shape": [10000, 784], "low": -10, "high": 10, "seed": 10}
        self.output_hist = output_hist

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        hist, prob = self.output_hist(np.array(outs[0]))
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.001), "hist: " + str(hist))


class TestRandintOpError(unittest.TestCase):
    def test_errors(self):
        main_prog = Program()
        start_prog = Program()
        with program_guard(main_prog, start_prog):

            def test_shape():
                shape = np.array([2, 3])
                paddle.randint(5, shape=shape, dtype='int32')

            self.assertRaises(TypeError, test_shape)

            def test_dtype():
                paddle.randint(5, shape=[32, 32], dtype='float32')

            self.assertRaises(TypeError, test_dtype)

            def test_low_high():
                paddle.randint(low=5, high=5, shape=[32, 32], dtype='int32')

            self.assertRaises(ValueError, test_low_high)


class TestRandintOp_attr_tensorlist(OpTest):
    def setUp(self):
        self.op_type = "randint"
        self.new_shape = (10000, 784)
        shape_tensor = []
        for index, ele in enumerate(self.new_shape):
            shape_tensor.append(("x" + str(index), np.ones(
                (1)).astype("int64") * ele))
        self.inputs = {'ShapeTensorList': shape_tensor}
        self.init_attrs()
        self.outputs = {"Out": np.zeros((10000, 784)).astype("int32")}

    def init_attrs(self):
        self.attrs = {"low": -10, "high": 10, "seed": 10}
        self.output_hist = output_hist

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        hist, prob = self.output_hist(np.array(outs[0]))
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.001), "hist: " + str(hist))


class TestRandint_attr_tensor(OpTest):
    def setUp(self):
        self.op_type = "randint"
        self.inputs = {"ShapeTensor": np.array([10000, 784]).astype("int64")}
        self.init_attrs()
        self.outputs = {"Out": np.zeros((10000, 784)).astype("int64")}

    def init_attrs(self):
        self.attrs = {"low": -10, "high": 10, "seed": 10}
        self.output_hist = output_hist

    def test_check_output(self):
        self.check_output_customized(self.verify_output)

    def verify_output(self, outs):
        hist, prob = self.output_hist(np.array(outs[0]))
        self.assertTrue(
            np.allclose(
                hist, prob, rtol=0, atol=0.001), "hist: " + str(hist))


# Test python API
class TestRandintAPI(unittest.TestCase):
    def test_api(self):
        startup_program = fluid.Program()
        train_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            # results are from [0, 5).
            output1 = paddle.randint(5)
            # shape is a list and dtype is 'int32'
            output2 = paddle.randint(
                low=-100, high=100, shape=[64, 64], dtype='int32')
            # shape is a tuple and dtype is 'int64'
            output3 = paddle.randint(
                low=-100, high=100, shape=(32, 32, 3), dtype='int64')
            # shape is a tensorlist and dtype is 'float32'
            dim_1 = fluid.layers.fill_constant([1], "int64", 32)
            dim_2 = fluid.layers.fill_constant([1], "int32", 50)
            output4 = paddle.randint(
                low=-100, high=100, shape=[dim_1, 5], dtype='int32')
            # shape is a tensor and dtype is 'float64'
            var_shape = fluid.data(name='var_shape', shape=[2], dtype="int64")
            output5 = paddle.randint(
                low=1, high=1000, shape=var_shape, dtype='int64')

            place = fluid.CPUPlace()
            if fluid.core.is_compiled_with_cuda():
                place = fluid.CUDAPlace(0)
            exe = fluid.Executor(place)

            exe.run(startup_program)
            outs = exe.run(
                train_program,
                feed={'var_shape': np.array([100, 100]).astype('int64')},
                fetch_list=[output1, output2, output3, output4, output5])


class TestRandintDygraphMode(unittest.TestCase):
    def test_check_output(self):
        with fluid.dygraph.guard():
            x = paddle.randint(10, shape=[10], dtype="int32")
            x_np = x.numpy()
            for i in range(10):
                self.assertTrue((x_np[i] >= 0 and x_np[i] < 10))


if __name__ == "__main__":
    unittest.main()
