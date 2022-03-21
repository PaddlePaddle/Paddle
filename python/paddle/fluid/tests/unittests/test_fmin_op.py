# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid.core as core
from op_test import OpTest

paddle.enable_static()


class ApiFMinTest(unittest.TestCase):
    """ApiFMinTest"""

    def setUp(self):
        """setUp"""
        if core.is_compiled_with_cuda():
            self.place = core.CUDAPlace(0)
        else:
            self.place = core.CPUPlace()

        self.input_x = np.random.rand(10, 15).astype("float32")
        self.input_y = np.random.rand(10, 15).astype("float32")
        self.input_z = np.random.rand(15).astype("float32")
        self.input_a = np.array([0, np.nan, np.nan]).astype('int64')
        self.input_b = np.array([2, np.inf, -np.inf]).astype('int64')
        self.input_c = np.array([4, 1, 3]).astype('int64')

        self.np_expected1 = np.fmin(self.input_x, self.input_y)
        self.np_expected2 = np.fmin(self.input_x, self.input_z)
        self.np_expected3 = np.fmin(self.input_a, self.input_c)
        self.np_expected4 = np.fmin(self.input_b, self.input_c)

    def test_static_api(self):
        """test_static_api"""
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            data_x = paddle.static.data("x", shape=[10, 15], dtype="float32")
            data_y = paddle.static.data("y", shape=[10, 15], dtype="float32")
            result_fmin = paddle.fmin(data_x, data_y)
            exe = paddle.static.Executor(self.place)
            res, = exe.run(feed={"x": self.input_x,
                                 "y": self.input_y},
                           fetch_list=[result_fmin])
        self.assertTrue(np.allclose(res, self.np_expected1))

        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            data_x = paddle.static.data("x", shape=[10, 15], dtype="float32")
            data_z = paddle.static.data("z", shape=[15], dtype="float32")
            result_fmin = paddle.fmin(data_x, data_z)
            exe = paddle.static.Executor(self.place)
            res, = exe.run(feed={"x": self.input_x,
                                 "z": self.input_z},
                           fetch_list=[result_fmin])
        self.assertTrue(np.allclose(res, self.np_expected2))

        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            data_a = paddle.static.data("a", shape=[3], dtype="int64")
            data_c = paddle.static.data("c", shape=[3], dtype="int64")
            result_fmin = paddle.fmin(data_a, data_c)
            exe = paddle.static.Executor(self.place)
            res, = exe.run(feed={"a": self.input_a,
                                 "c": self.input_c},
                           fetch_list=[result_fmin])
        self.assertTrue(np.allclose(res, self.np_expected3))

        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            data_b = paddle.static.data("b", shape=[3], dtype="int64")
            data_c = paddle.static.data("c", shape=[3], dtype="int64")
            result_fmin = paddle.fmin(data_b, data_c)
            exe = paddle.static.Executor(self.place)
            res, = exe.run(feed={"b": self.input_b,
                                 "c": self.input_c},
                           fetch_list=[result_fmin])
        self.assertTrue(np.allclose(res, self.np_expected4))

    def test_dynamic_api(self):
        """test_dynamic_api"""
        paddle.disable_static()
        x = paddle.to_tensor(self.input_x)
        y = paddle.to_tensor(self.input_y)
        z = paddle.to_tensor(self.input_z)

        a = paddle.to_tensor(self.input_a)
        b = paddle.to_tensor(self.input_b)
        c = paddle.to_tensor(self.input_c)

        res = paddle.fmin(x, y)
        res = res.numpy()
        self.assertTrue(np.allclose(res, self.np_expected1))

        # test broadcast
        res = paddle.fmin(x, z)
        res = res.numpy()
        self.assertTrue(np.allclose(res, self.np_expected2))

        res = paddle.fmin(a, c)
        res = res.numpy()
        self.assertTrue(np.allclose(res, self.np_expected3))

        res = paddle.fmin(b, c)
        res = res.numpy()
        self.assertTrue(np.allclose(res, self.np_expected4))


class TestElementwiseFminOp(OpTest):
    """TestElementwiseFminOp"""

    def setUp(self):
        """setUp"""
        self.op_type = "elementwise_fmin"
        # If x and y have the same value, the min() is not differentiable.
        # So we generate test data by the following method
        # to avoid them being too close to each other.
        x = np.random.uniform(0.1, 1, [13, 17]).astype("float64")
        sgn = np.random.choice([-1, 1], [13, 17]).astype("float64")
        y = x + sgn * np.random.uniform(0.1, 1, [13, 17]).astype("float64")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.fmin(self.inputs['X'], self.inputs['Y'])}

    def test_check_output(self):
        """test_check_output"""
        self.check_output()

    def test_check_grad_normal(self):
        """test_check_grad_normal"""
        self.check_grad(['X', 'Y'], 'Out')

    def test_check_grad_ingore_x(self):
        """test_check_grad_ingore_x"""
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.005, no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        """test_check_grad_ingore_y"""
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.005, no_grad_set=set('Y'))


class TestElementwiseFmin2Op(OpTest):
    """TestElementwiseFmin2Op"""

    def setUp(self):
        """setUp"""
        self.op_type = "elementwise_fmin"
        # If x and y have the same value, the min() is not differentiable.
        # So we generate test data by the following method
        # to avoid them being too close to each other.
        x = np.random.uniform(0.1, 1, [13, 17]).astype("float64")
        sgn = np.random.choice([-1, 1], [13, 17]).astype("float64")
        y = x + sgn * np.random.uniform(0.1, 1, [13, 17]).astype("float64")

        y[2, 10:] = np.nan
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.fmin(self.inputs['X'], self.inputs['Y'])}

    def test_check_output(self):
        """test_check_output"""
        self.check_output()

    def test_check_grad_normal(self):
        """test_check_grad_normal"""
        self.check_grad(['X', 'Y'], 'Out')

    def test_check_grad_ingore_x(self):
        """test_check_grad_ingore_x"""
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.005, no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        """test_check_grad_ingore_y"""
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.005, no_grad_set=set('Y'))


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
