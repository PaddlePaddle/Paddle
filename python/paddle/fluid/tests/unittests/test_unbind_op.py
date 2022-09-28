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
import numpy as np
from op_test import OpTest, convert_float_to_uint16
import paddle
import paddle.fluid as fluid
import paddle.tensor as tensor
from paddle.fluid import compiler, Program, program_guard, core
from paddle.fluid.framework import _test_eager_guard


class TestUnbind(unittest.TestCase):

    def test_unbind(self):

        x_1 = fluid.data(shape=[2, 3], dtype='float32', name='x_1')
        [out_0, out_1] = tensor.unbind(input=x_1, axis=0)
        input_1 = np.random.random([2, 3]).astype("float32")
        axis = fluid.data(shape=[1], dtype='int32', name='axis')
        exe = fluid.Executor(place=fluid.CPUPlace())

        [res_1, res_2] = exe.run(fluid.default_main_program(),
                                 feed={
                                     "x_1": input_1,
                                     "axis": 0
                                 },
                                 fetch_list=[out_0, out_1])

        assert np.array_equal(res_1, input_1[0, 0:100])
        assert np.array_equal(res_2, input_1[1, 0:100])

    def test_unbind_dygraph(self):
        with fluid.dygraph.guard():
            np_x = np.random.random([2, 3]).astype("float32")
            x = paddle.to_tensor(np_x)
            x.stop_gradient = False
            [res_1, res_2] = paddle.unbind(x, 0)
            np.testing.assert_array_equal(res_1, np_x[0, 0:100])
            np.testing.assert_array_equal(res_2, np_x[1, 0:100])

            out = paddle.add_n([res_1, res_2])

            np_grad = np.ones(x.shape, np.float32)
            out.backward()
            np.testing.assert_array_equal(x.grad.numpy(), np_grad)

    def test_unbind_dygraph_final_state(self):
        with _test_eager_guard():
            self.test_unbind_dygraph()


class TestLayersUnbind(unittest.TestCase):

    def test_layers_unbind(self):

        x_1 = fluid.data(shape=[2, 3], dtype='float32', name='x_1')
        [out_0, out_1] = fluid.layers.unbind(input=x_1, axis=0)
        input_1 = np.random.random([2, 3]).astype("float32")
        axis = fluid.data(shape=[1], dtype='int32', name='axis')
        exe = fluid.Executor(place=fluid.CPUPlace())

        [res_1, res_2] = exe.run(fluid.default_main_program(),
                                 feed={
                                     "x_1": input_1,
                                     "axis": 0
                                 },
                                 fetch_list=[out_0, out_1])

        assert np.array_equal(res_1, input_1[0, 0:100])
        assert np.array_equal(res_2, input_1[1, 0:100])


class TestUnbindOp(OpTest):

    def initParameters(self):
        pass

    def outReshape(self):
        pass

    def setAxis(self):
        pass

    def setUp(self):
        self._set_op_type()
        self.dtype = self.get_dtype()
        self.axis = 0
        self.num = 3
        self.initParameters()
        x = np.arange(12).reshape(3, 2, 2).astype(self.dtype)
        self.out = np.split(x, self.num, self.axis)
        self.outReshape()
        self.inputs = {'X': x}
        self.attrs = {'axis': self.axis}
        self.setAxis()
        self.outputs = {'Out': [('out%d' % i, self.out[i]) \
            for i in range(len(self.out))]}

    def get_dtype(self):
        return "float64"

    def _set_op_type(self):
        self.op_type = "unbind"

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], ['out0', 'out1', 'out2'])


class TestUnbindOp1(TestUnbindOp):

    def initParameters(self):
        self.axis = 1
        self.num = 2

    def test_check_grad(self):
        self.check_grad(['X'], ['out0', 'out1'])

    def outReshape(self):
        self.out[0] = self.out[0].reshape((3, 2))
        self.out[1] = self.out[1].reshape((3, 2))


class TestUnbindOp2(TestUnbindOp):

    def initParameters(self):
        self.axis = 2
        self.num = 2

    def test_check_grad(self):
        self.check_grad(['X'], ['out0', 'out1'])

    def outReshape(self):
        self.out[0] = self.out[0].reshape((3, 2))
        self.out[1] = self.out[1].reshape((3, 2))


class TestUnbindOp3(TestUnbindOp):

    def initParameters(self):
        self.axis = 2
        self.num = 2

    def setAxis(self):
        self.attrs = {'axis': -1}

    def test_check_grad(self):
        self.check_grad(['X'], ['out0', 'out1'])

    def outReshape(self):
        self.out[0] = self.out[0].reshape((3, 2))
        self.out[1] = self.out[1].reshape((3, 2))


class TestUnbindOp4(TestUnbindOp):

    def initParameters(self):
        self.axis = 1
        self.num = 2

    def setAxis(self):
        self.attrs = {'axis': -2}

    def test_check_grad(self):
        self.check_grad(['X'], ['out0', 'out1'])

    def outReshape(self):
        self.out[0] = self.out[0].reshape((3, 2))
        self.out[1] = self.out[1].reshape((3, 2))


class TestUnbindBF16Op(OpTest):

    def setUp(self):
        self._set_op_type()
        self.python_api = paddle.unbind
        self.dtype = self.get_dtype()
        self.axis = 0
        self.num = 3
        x = np.arange(12).reshape(3, 2, 2).astype(self.dtype)
        self.out = np.split(x, self.num, self.axis)
        self.inputs = {'X': convert_float_to_uint16(x)}
        self.attrs = {'axis': self.axis}
        self.outputs = {'Out': [('out%d' % i, convert_float_to_uint16(self.out[i])) \
            for i in range(len(self.out))]}

    def get_dtype(self):
        return np.uint16

    def _set_op_type(self):
        self.op_type = "unbind"

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        pass


class TestUnbindAxisError(unittest.TestCase):

    def test_errors(self):
        with program_guard(Program(), Program()):
            x = fluid.data(shape=[2, 3], dtype='float32', name='x')

            def test_table_Variable():
                tensor.unbind(input=x, axis=2.0)

            self.assertRaises(TypeError, test_table_Variable)


if __name__ == '__main__':
    unittest.main()
