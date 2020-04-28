#Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.core as core
from op_test import OpTest
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid.op import Operator
from paddle.fluid.backward import append_backward


class TestWhereOp(OpTest):
    def setUp(self):
        self.op_type = "where"
        self.init_config()
        self.inputs = {'Condition': self.cond, 'X': self.x, 'Y': self.y}
        self.outputs = {'Out': np.where(self.cond, self.x, self.y)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'Y'], 'Out')

    def init_config(self):
        self.x = np.random.uniform(-3, 5, (100)).astype("float64")
        self.y = np.random.uniform(-3, 5, (100)).astype("float64")
        self.cond = np.zeros((100)).astype("bool")


class TestWhereOp2(TestWhereOp):
    def init_config(self):
        self.x = np.random.uniform(-5, 5, (60, 2)).astype("float64")
        self.y = np.random.uniform(-5, 5, (60, 2)).astype("float64")
        self.cond = np.ones((60, 2)).astype("bool")


class TestWhereOp3(TestWhereOp):
    def init_config(self):
        self.x = np.random.uniform(-3, 5, (20, 2, 4)).astype("float64")
        self.y = np.random.uniform(-3, 5, (20, 2, 4)).astype("float64")
        self.cond = np.array(np.random.randint(2, size=(20, 2, 4)), dtype=bool)


class TestWhereAPI(unittest.TestCase):
    def setUp(self):
        self.init_data()

    def init_data(self):
        self.shape = [10, 15]
        self.cond = np.array(np.random.randint(2, size=self.shape), dtype=bool)
        self.x = np.random.uniform(-2, 3, self.shape).astype(np.float32)
        self.y = np.random.uniform(-2, 3, self.shape).astype(np.float32)
        self.out = np.where(self.cond, self.x, self.y)

    def ref_x_backward(self, dout):
        return np.where(self.cond == True, dout, 0)

    def ref_y_backward(self, dout):
        return np.where(self.cond == False, dout, 0)

    def test_api(self, use_cuda=False):
        for x_stop_gradient in [False, True]:
            for y_stop_gradient in [False, True]:
                with fluid.program_guard(Program(), Program()):
                    cond = fluid.layers.data(
                        name='cond', shape=self.shape, dtype='bool')
                    x = fluid.layers.data(
                        name='x', shape=self.shape, dtype='float32')
                    y = fluid.layers.data(
                        name='y', shape=self.shape, dtype='float32')
                    x.stop_gradient = x_stop_gradient
                    y.stop_gradient = y_stop_gradient
                    result = paddle.where(cond, x, y)
                    append_backward(layers.mean(result))

                    for use_cuda in [False, True]:
                        if use_cuda and not fluid.core.is_compiled_with_cuda():
                            break
                        place = fluid.CUDAPlace(
                            0) if use_cuda else fluid.CPUPlace()
                        exe = fluid.Executor(place)
                        fetch_list = [result, result.grad_name]
                        if x_stop_gradient is False:
                            fetch_list.append(x.grad_name)
                        if y_stop_gradient is False:
                            fetch_list.append(y.grad_name)
                        out = exe.run(
                            fluid.default_main_program(),
                            feed={'cond': self.cond,
                                  'x': self.x,
                                  'y': self.y},
                            fetch_list=fetch_list)
                        assert np.array_equal(out[0], self.out)
                        if x_stop_gradient is False:
                            assert np.array_equal(out[2],
                                                  self.ref_x_backward(out[1]))
                            if y.stop_gradient is False:
                                assert np.array_equal(
                                    out[3], self.ref_y_backward(out[1]))
                        elif y.stop_gradient is False:
                            assert np.array_equal(out[2],
                                                  self.ref_y_backward(out[1]))

    def test_api_broadcast(self, use_cuda=False):
        main_program = Program()
        with fluid.program_guard(main_program):
            x = fluid.layers.data(name='x', shape=[4, 1], dtype='float32')
            y = fluid.layers.data(name='y', shape=[4, 2], dtype='float32')
            x_i = np.array([[0.9383, 0.1983, 3.2, 1.2]]).astype("float32")
            y_i = np.array([[1.0, 1.0, 1.0, 1.0],
                            [1.0, 1.0, 1.0, 1.0]]).astype("float32")
            result = paddle.where(x > 1, x=x, y=y)

            for use_cuda in [False, True]:
                if use_cuda and not fluid.core.is_compiled_with_cuda():
                    return
                place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
                exe = fluid.Executor(place)
                out = exe.run(fluid.default_main_program(),
                              feed={'x': x_i,
                                    'y': y_i},
                              fetch_list=[result])
                assert np.array_equal(out[0], np.where(x_i > 1, x_i, y_i))


class TestWhereDygraphAPI(unittest.TestCase):
    def test_api(self):
        with fluid.dygraph.guard():
            x_i = np.array([0.9383, 0.1983, 3.2, 1.2]).astype("float64")
            y_i = np.array([1.0, 1.0, 1.0, 1.0]).astype("float64")
            cond_i = np.array([False, False, True, True]).astype("bool")
            x = fluid.dygraph.to_variable(x_i)
            y = fluid.dygraph.to_variable(y_i)
            cond = fluid.dygraph.to_variable(cond_i)
            out = paddle.where(cond, x, y)
            assert np.array_equal(out.numpy(), np.where(cond_i, x_i, y_i))


class TestWhereOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            x_i = np.array([0.9383, 0.1983, 3.2, 1.2]).astype("float64")
            y_i = np.array([1.0, 1.0, 1.0, 1.0]).astype("float64")
            cond_i = np.array([False, False, True, True]).astype("bool")

            def test_Variable():
                paddle.where(cond_i, x_i, y_i)

            self.assertRaises(TypeError, test_Variable)

            def test_type():
                x = fluid.layers.data(name='x', shape=[4], dtype='bool')
                y = fluid.layers.data(name='y', shape=[4], dtype='float16')
                cond = fluid.layers.data(name='cond', shape=[4], dtype='int32')
                paddle.where(cond, x, y)

            self.assertRaises(TypeError, test_type)


if __name__ == '__main__':
    unittest.main()
