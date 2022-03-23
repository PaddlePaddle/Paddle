#  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function
import unittest
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
from op_test import OpTest, skip_check_grad_ci, convert_float_to_uint16


class ElementwiseDivOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.python_api = paddle.divide
        self.dtype = np.float64
        self.init_dtype()
        """ Warning
        CPU gradient check error!
        'X': np.random.random((32,84)).astype("float32"),
        'Y': np.random.random((32,84)).astype("float32")
        """
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype),
            'Y': np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        }
        self.outputs = {'Out': np.divide(self.inputs['X'], self.inputs['Y'])}

    def check_eager(self):
        return (self.use_mkldnn == False and self.axis == -1)

    def test_check_output(self):
        self.check_output(check_eager=False)

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=0.05)

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.05, no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.05, no_grad_set=set('Y'))

    def init_dtype(self):
        pass


@unittest.skipIf(
    not core.is_compiled_with_cuda() or core.cudnn_version() < 8100,
    "core is not compiled with CUDA and cudnn version need larger than 8.1.0")
class TestElementwiseDivOpBF16(OpTest):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.dtype = np.uint16

        x = np.random.uniform(0.1, 1, [12, 13]).astype(np.float32)
        y = np.random.uniform(0.1, 1, [12, 13]).astype(np.float32)

        out = np.divide(x, y)

        self.inputs = {
            'X': convert_float_to_uint16(x),
            'Y': convert_float_to_uint16(y)
        }
        self.outputs = {'Out': convert_float_to_uint16(out)}

    def test_check_output(self):
        place = core.CUDAPlace(0)
        self.check_output_with_place(place)

    def test_check_grad_normal(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, ['X', 'Y'], 'Out')

    def test_check_grad_ingore_x(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, ['Y'], 'Out', no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, ['X'], 'Out', no_grad_set=set('Y'))


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast.")
class TestElementwiseDivOp_scalar(ElementwiseDivOp):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [20, 3, 4]).astype(np.float64),
            'Y': np.random.uniform(0.1, 1, [1]).astype(np.float64)
        }
        self.outputs = {'Out': self.inputs['X'] / self.inputs['Y']}


class TestElementwiseDivOp_Vector(ElementwiseDivOp):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [100]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [100]).astype("float64")
        }
        self.outputs = {'Out': np.divide(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseDivOp_broadcast_0(ElementwiseDivOp):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [100, 3, 4]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [100]).astype("float64")
        }

        self.attrs = {'axis': 0}
        self.outputs = {
            'Out':
            np.divide(self.inputs['X'], self.inputs['Y'].reshape(100, 1, 1))
        }


class TestElementwiseDivOp_broadcast_1(ElementwiseDivOp):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 100, 4]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [100]).astype("float64")
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out':
            np.divide(self.inputs['X'], self.inputs['Y'].reshape(1, 100, 1))
        }


class TestElementwiseDivOp_broadcast_2(ElementwiseDivOp):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 100]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [100]).astype("float64")
        }

        self.outputs = {
            'Out':
            np.divide(self.inputs['X'], self.inputs['Y'].reshape(1, 1, 100))
        }


class TestElementwiseDivOp_broadcast_3(ElementwiseDivOp):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 10, 12, 5]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [10, 12]).astype("float64")
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out':
            np.divide(self.inputs['X'], self.inputs['Y'].reshape(1, 10, 12, 1))
        }


class TestElementwiseDivOp_broadcast_4(ElementwiseDivOp):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 50]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [2, 1, 50]).astype("float64")
        }
        self.outputs = {'Out': np.divide(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseDivOp_broadcast_5(ElementwiseDivOp):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 4, 20]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [2, 3, 1, 20]).astype("float64")
        }
        self.outputs = {'Out': np.divide(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseDivOp_commonuse_1(ElementwiseDivOp):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 100]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [1, 1, 100]).astype("float64"),
        }
        self.outputs = {'Out': np.divide(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseDivOp_commonuse_2(ElementwiseDivOp):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [30, 3, 1, 5]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [30, 1, 4, 1]).astype("float64"),
        }
        self.outputs = {'Out': np.divide(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseDivOp_xsize_lessthan_ysize(ElementwiseDivOp):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [10, 12]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [2, 3, 10, 12]).astype("float64"),
        }

        self.attrs = {'axis': 2}

        self.outputs = {'Out': np.divide(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseDivOp_INT(OpTest):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.dtype = np.int32
        self.init_dtype()
        self.inputs = {
            'X': np.random.randint(
                1, 5, size=[13, 17]).astype(self.dtype),
            'Y': np.random.randint(
                1, 5, size=[13, 17]).astype(self.dtype)
        }
        self.outputs = {'Out': self.inputs['X'] // self.inputs['Y']}

    def test_check_output(self):
        self.check_output()

    def init_dtype(self):
        pass


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestElementwiseDivOpFp16(ElementwiseDivOp):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out', max_relative_error=1)

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=1, no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=1, no_grad_set=set('Y'))


class TestElementwiseDivBroadcast(unittest.TestCase):
    def test_shape_with_batch_sizes(self):
        with fluid.program_guard(fluid.Program()):
            x_var = fluid.data(
                name='x', dtype='float32', shape=[None, 3, None, None])
            one = 2.
            out = one / x_var
            exe = fluid.Executor(fluid.CPUPlace())
            x = np.random.uniform(0.1, 0.6, (1, 3, 32, 32)).astype("float32")
            out_result, = exe.run(feed={'x': x}, fetch_list=[out])
            self.assertEqual((out_result == (2 / x)).all(), True)


class TestDivideOp(unittest.TestCase):
    def test_name(self):
        with fluid.program_guard(fluid.Program()):
            x = fluid.data(name="x", shape=[2, 3], dtype="float32")
            y = fluid.data(name='y', shape=[2, 3], dtype='float32')

            y_1 = paddle.divide(x, y, name='div_res')
            self.assertEqual(('div_res' in y_1.name), True)

    def test_dygraph(self):
        with fluid.dygraph.guard():
            np_x = np.array([2, 3, 4]).astype('float64')
            np_y = np.array([1, 5, 2]).astype('float64')
            x = paddle.to_tensor(np_x)
            y = paddle.to_tensor(np_y)
            z = paddle.divide(x, y)
            np_z = z.numpy()
            z_expected = np.array([2., 0.6, 2.])
            self.assertEqual((np_z == z_expected).all(), True)


class TestComplexElementwiseDivOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_div"
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
            (2, 3, 4, 5)).astype(self.dtype) + 1J * np.random.random(
                (2, 3, 4, 5)).astype(self.dtype)
        self.y = np.random.random(
            (2, 3, 4, 5)).astype(self.dtype) + 1J * np.random.random(
                (2, 3, 4, 5)).astype(self.dtype)
        self.out = self.x / self.y

    def init_grad_input_output(self):
        self.grad_out = np.ones((2, 3, 4, 5), self.dtype) + 1J * np.ones(
            (2, 3, 4, 5), self.dtype)
        self.grad_x = self.grad_out / np.conj(self.y)
        self.grad_y = -self.grad_out * np.conj(self.x / self.y / self.y)

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            user_defined_grads=[self.grad_x, self.grad_y],
            user_defined_grad_outputs=[self.grad_out])

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'],
            'Out',
            no_grad_set=set("X"),
            user_defined_grads=[self.grad_y],
            user_defined_grad_outputs=[self.grad_out])

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'],
            'Out',
            no_grad_set=set('Y'),
            user_defined_grads=[self.grad_x],
            user_defined_grad_outputs=[self.grad_out])


class TestRealComplexElementwiseDivOp(TestComplexElementwiseDivOp):
    def init_input_output(self):
        self.x = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.y = np.random.random(
            (2, 3, 4, 5)).astype(self.dtype) + 1J * np.random.random(
                (2, 3, 4, 5)).astype(self.dtype)
        self.out = self.x / self.y

    def init_grad_input_output(self):
        self.grad_out = np.ones((2, 3, 4, 5), self.dtype) + 1J * np.ones(
            (2, 3, 4, 5), self.dtype)
        self.grad_x = np.real(self.grad_out / np.conj(self.y))
        self.grad_y = -self.grad_out * np.conj(self.x / self.y / self.y)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
