#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import unittest
import numpy as np
from op_test import OpTest, skip_check_grad_ci
from paddle.fluid.op import Operator
from paddle.fluid import compiler, Program, program_guard


class DotOp(OpTest):

    def setUp(self):
        self.op_type = "dot"
        self.python_api = paddle.dot
        self.init_dtype()
        self.init_input_output()

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y)
        }
        self.outputs = {'Out': self.out}
        self.attrs = {}

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad_normal(self):
        if core.is_compiled_with_rocm():
            self.check_grad(
                ['X', 'Y'],
                'Out',
                user_defined_grads=[self.inputs['Y'], self.inputs['X']],
                check_eager=True)
        else:
            self.check_grad(['X', 'Y'], 'Out', check_eager=True)

    def test_check_grad_ingore_x(self):
        if core.is_compiled_with_rocm():
            self.check_grad(['Y'],
                            'Out',
                            no_grad_set=set("X"),
                            user_defined_grads=[self.inputs['X']],
                            check_eager=True)
        else:
            self.check_grad(['Y'],
                            'Out',
                            no_grad_set=set("X"),
                            check_eager=True)

    def test_check_grad_ingore_y(self):
        if core.is_compiled_with_rocm():
            self.check_grad(['X'],
                            'Out',
                            no_grad_set=set('Y'),
                            user_defined_grads=[self.inputs['Y']],
                            check_eager=True)
        else:
            self.check_grad(['X'],
                            'Out',
                            no_grad_set=set('Y'),
                            check_eager=True)

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [121]).astype(self.dtype)
        self.y = np.random.uniform(1, 3, [121]).astype(self.dtype)
        self.out = np.dot(self.x, self.y)

    def init_dtype(self):
        self.dtype = np.float64


class DotOpBatch(DotOp):

    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1,
                                   [132]).astype(self.dtype).reshape([11, 12])
        self.y = np.random.uniform(1, 3,
                                   [132]).astype(self.dtype).reshape([11, 12])
        self.out = np.sum(self.x * self.y, axis=1).reshape([11, 1])

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')

    def test_check_grad_ingore_x(self):
        self.check_grad(['Y'], 'Out', no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(['X'], 'Out', no_grad_set=set('Y'))


class TestDotOpError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):

            # the input dtype of elementwise_mul must be float16 or float32 or float64 or int32 or int64
            # float16 only can be set on GPU place
            x1 = fluid.layers.data(name='x1', shape=[120], dtype="uint8")
            y1 = fluid.layers.data(name='y1', shape=[120], dtype="uint8")
            self.assertRaises(Exception, paddle.dot, x1, y1)

            x2 = fluid.layers.data(name='x2', shape=[2, 3], dtype="float32")
            y2 = fluid.layers.data(name='y2', shape=[2, 3], dtype="float32")
            self.assertRaises(Exception, paddle.dot, x2, y2)

            x3 = fluid.layers.data(name='x3', shape=[3], dtype="float32")
            y3 = fluid.layers.data(name='y3', shape=[2, 3], dtype="float32")
            self.assertRaises(Exception, paddle.dot, x2, y3)


class TestDygraph(unittest.TestCase):

    def test_dygraph(self):
        with fluid.dygraph.guard():
            x1 = fluid.dygraph.to_variable(np.array([1, 3]).astype(np.float32))
            y1 = fluid.dygraph.to_variable(np.array([2, 5]).astype(np.float32))
            np.testing.assert_allclose(paddle.dot(x1, y1).numpy(),
                                       np.array([17]),
                                       rtol=1e-05)

            x1 = fluid.dygraph.to_variable(
                np.array([[1, 3], [3, 5]]).astype(np.float32))
            y1 = fluid.dygraph.to_variable(
                np.array([[2, 5], [6, 8]]).astype(np.float32))
            np.testing.assert_array_equal(
                paddle.dot(x1, y1).numpy(), np.array([[17], [58]]))


class TestComplexDotOp(OpTest):

    def setUp(self):
        self.op_type = "dot"
        self.python_api = paddle.dot
        self.init_base_dtype()
        self.init_input_output()
        self.init_grad_input_output()

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y)
        }
        self.attrs = {'axis': -1, 'use_mkldnn': False}
        self.outputs = {'Out': self.out}

    def init_base_dtype(self):
        self.dtype = np.float64

    def init_input_output(self):
        self.x = np.random.random(100).astype(
            self.dtype) + 1J * np.random.random(100).astype(self.dtype)
        self.y = np.random.random(100).astype(
            self.dtype) + 1J * np.random.random(100).astype(self.dtype)
        self.out = np.dot(self.x, self.y)

    def init_grad_input_output(self):
        self.grad_out = np.ones(1, self.dtype) + 1J * np.ones(1, self.dtype)
        self.grad_x = self.grad_out * np.conj(self.y)
        self.grad_y = self.grad_out * np.conj(self.x)

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'],
                        'Out',
                        user_defined_grads=[self.grad_x, self.grad_y],
                        user_defined_grad_outputs=[self.grad_out],
                        check_eager=True)

    def test_check_grad_ingore_x(self):
        self.check_grad(['Y'],
                        'Out',
                        no_grad_set=set("X"),
                        user_defined_grads=[self.grad_y],
                        user_defined_grad_outputs=[self.grad_out],
                        check_eager=True)

    def test_check_grad_ingore_y(self):
        self.check_grad(['X'],
                        'Out',
                        no_grad_set=set('Y'),
                        user_defined_grads=[self.grad_x],
                        user_defined_grad_outputs=[self.grad_out],
                        check_eager=True)


class TestComplexDotOp2D(OpTest):

    def setUp(self):
        self.op_type = "dot"
        self.init_base_dtype()
        self.init_input_output()
        self.init_grad_input_output()

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y)
        }
        self.attrs = {'axis': -1, 'use_mkldnn': False}
        self.outputs = {'Out': self.out}

    def init_base_dtype(self):
        self.dtype = np.float64

    def init_input_output(self):
        self.x = np.random.random(
            (2, 100)).astype(self.dtype) + 1J * np.random.random(
                (2, 100)).astype(self.dtype)
        self.y = np.random.random(
            (2, 100)).astype(self.dtype) + 1J * np.random.random(
                (2, 100)).astype(self.dtype)
        self.out = np.diag(np.dot(self.x, self.y.T)).reshape(-1, 1)

    def init_grad_input_output(self):
        self.grad_out = np.ones((2, 1), self.dtype) + 1J * np.ones(
            (2, 1), self.dtype)
        self.grad_x = self._get_grad(self.grad_out, self.y)
        self.grad_y = self._get_grad(self.grad_out, self.x)

    def _get_grad(self, grad_out, input):
        grad = np.empty((0, input.shape[1]))
        for i in range(grad_out.shape[0]):
            grad = np.append(grad, [grad_out[i] * np.conj(input[i])], axis=0)
        return grad

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'],
                        'Out',
                        user_defined_grads=[self.grad_x, self.grad_y],
                        user_defined_grad_outputs=[self.grad_out])

    def test_check_grad_ingore_x(self):
        self.check_grad(['Y'],
                        'Out',
                        no_grad_set=set("X"),
                        user_defined_grads=[self.grad_y],
                        user_defined_grad_outputs=[self.grad_out])

    def test_check_grad_ingore_y(self):
        self.check_grad(['X'],
                        'Out',
                        no_grad_set=set('Y'),
                        user_defined_grads=[self.grad_x],
                        user_defined_grad_outputs=[self.grad_out])


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
