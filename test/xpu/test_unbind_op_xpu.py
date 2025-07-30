#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from get_test_cover_info import (
    XPUOpTestWrapper,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import convert_uint16_to_float
from op_test_xpu import XPUOpTest

import paddle
from paddle import base, tensor
from paddle.base import Program, program_guard

paddle.enable_static()


class XPUTestUnbindOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'unbind'
        self.use_dynamic_create_class = False

    class TestUnbind(unittest.TestCase):
        def test_unbind(self):
            with program_guard(Program(), Program()):
                self.dtype = self.in_type
                self.place = paddle.XPUPlace(0)
                x_1 = paddle.static.data(
                    shape=[2, 3], dtype=self.dtype, name='x_1'
                )
                [out_0, out_1] = tensor.unbind(input=x_1, axis=0)
                input_1 = np.random.random([2, 3]).astype(self.dtype)
                axis = paddle.static.data(shape=[], dtype='int32', name='axis')
                exe = base.Executor(place=self.place)

                [res_1, res_2] = exe.run(
                    base.default_main_program(),
                    feed={"x_1": input_1, "axis": 0},
                    fetch_list=[out_0, out_1],
                )

                np.testing.assert_array_equal(res_1, input_1[0, 0:100])
                np.testing.assert_array_equal(res_2, input_1[1, 0:100])

        def test_unbind_dygraph(self):
            with base.dygraph.guard():
                self.dtype = self.in_type
                self.place = paddle.XPUPlace(0)
                np_x = np.random.random([2, 3]).astype(self.dtype)
                if self.dtype == np.uint16:
                    np_x = convert_uint16_to_float(np_x)
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
            self.test_unbind_dygraph()

    class TestLayersUnbind(unittest.TestCase):
        def test_layers_unbind(self):
            with program_guard(Program(), Program()):
                self.dtype = self.in_type
                self.place = paddle.XPUPlace(0)
                x_1 = paddle.static.data(
                    shape=[2, 3], dtype=self.dtype, name='x_1'
                )
                [out_0, out_1] = paddle.unbind(input=x_1, axis=0)
                input_1 = np.random.random([2, 3]).astype(self.dtype)
                axis = paddle.static.data(shape=[], dtype='int32', name='axis')
                exe = base.Executor(place=self.place)

                [res_1, res_2] = exe.run(
                    base.default_main_program(),
                    feed={"x_1": input_1, "axis": 0},
                    fetch_list=[out_0, out_1],
                )

                np.testing.assert_array_equal(res_1, input_1[0, 0:100])
                np.testing.assert_array_equal(res_2, input_1[1, 0:100])

    class TestUnbindOp(XPUOpTest):
        def initParameters(self):
            pass

        def outReshape(self):
            self.out[0] = self.out[0].reshape((2, 2))
            self.out[1] = self.out[1].reshape((2, 2))
            self.out[2] = self.out[2].reshape((2, 2))

        def setAxis(self):
            pass

        def setUp(self):
            self._set_op_type()
            self.dtype = self.in_type
            self.place = paddle.XPUPlace(0)
            self.axis = 0
            self.num = 3
            self.initParameters()
            x = np.arange(12).reshape(3, 2, 2).astype(self.dtype)
            self.out = np.split(x, self.num, self.axis)
            self.outReshape()
            self.inputs = {'X': x}
            self.attrs = {'axis': self.axis}
            self.setAxis()
            self.outputs = {
                'Out': [(f'out{i}', self.out[i]) for i in range(len(self.out))]
            }

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

    class TestUnbindAxisError(unittest.TestCase):
        def test_errors(self):
            with program_guard(Program(), Program()):
                self.dtype = self.in_type
                self.place = paddle.XPUPlace(0)
                x = paddle.static.data(shape=[2, 3], dtype=self.dtype, name='x')

                def test_table_Variable():
                    tensor.unbind(input=x, axis=2.0)

                self.assertRaises(TypeError, test_table_Variable)

                def test_invalid_axis():
                    tensor.unbind(input=x, axis=2)

                self.assertRaises(ValueError, test_invalid_axis)


support_types = get_xpu_op_support_types('unbind')
for stype in support_types:
    create_test_class(globals(), XPUTestUnbindOP, stype)


if __name__ == '__main__':
    unittest.main()
