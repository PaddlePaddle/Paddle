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
from op_test import OpTest, skip_check_grad_ci


class ElementwiseDivOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_div"
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

    def test_check_output(self):
        self.check_output()

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


class TestDivideAPI(unittest.TestCase):
    def setUp(self):
        paddle.set_default_dtype("float64")
        self.places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            self.places.append(fluid.CUDAPlace(0))

    def check_static_result(self, place):
        # rule 1
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            x = fluid.data(name="x", shape=[3], dtype="float64")
            y = np.array([1, 2, 3])
            self.assertRaises(TypeError, paddle.divide, x=x, y=y)

        # rule 2: both the inputs are not Tensor
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            x = 2
            y = 4
            res = paddle.divide(x, y)
            exe = fluid.Executor(place)
            np_z = exe.run(fluid.default_main_program(),
                           feed={},
                           fetch_list=[res])
            self.assertEqual(np_z[0] == 0.5, True)

        # rule 3: 
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            x = fluid.data(name="x", shape=[3], dtype="float64")
            y = fluid.data(name="y", shape=[3], dtype="float32")
            self.assertRaises(TypeError, paddle.divide, x=x, y=y)

        # rule 4: x is Tensor, y is scalar
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            x = fluid.data(name="x", shape=[3], dtype="float64")
            y = 2
            exe = fluid.Executor(place)
            res = x / y
            np_z = exe.run(fluid.default_main_program(),
                           feed={"x": np.array([2, 3, 4]).astype('float64')},
                           fetch_list=[res])
            z_expected = np.array([1., 1.5, 2.])
            self.assertEqual((np_z[0] == z_expected).all(), True)

        # rule 5: y is Tensor, x is scalar
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            x = fluid.data(name="x", shape=[3], dtype="float64")
            y = 2
            exe = fluid.Executor(place)
            res = y / x
            np_z = exe.run(fluid.default_main_program(),
                           feed={"x": np.array([2, 8, 4]).astype('float64')},
                           fetch_list=[res])
            z_expected = np.array([1., 0.25, 0.5])
            self.assertEqual((np_z[0] == z_expected).all(), True)

        # rule 6: y is Tensor, x is Tensor
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            x = fluid.data(name="x", shape=[3], dtype="float64")
            y = fluid.data(name="y", shape=[3], dtype="float64")
            exe = fluid.Executor(place)
            res = x / y
            np_z = exe.run(fluid.default_main_program(),
                           feed={
                               "x": np.array([2, 3, 4]).astype('float64'),
                               "y": np.array([1, 5, 2]).astype('float64')
                           },
                           fetch_list=[res])
            z_expected = np.array([2., 0.6, 2.])
            self.assertEqual((np_z[0] == z_expected).all(), True)

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            with fluid.dygraph.guard(place):
                # rule 1 : avoid numpy.ndarray
                np_x = np.array([2, 3, 4])
                np_y = np.array([1, 5, 2])
                x = paddle.to_tensor(np_x)
                self.assertRaises(TypeError, paddle.divide, x=x, y=np_y)

                # rule 2: both the inputs are not Tensor
                z = paddle.divide(3, 2)
                self.assertEqual(z.numpy()[0] == 1.5, True)

                # rule 3: both the inputs are Tensor
                np_x = np.array([2, 3, 4])
                np_y = np.array([1, 5, 2])
                x = paddle.to_tensor(np_x, dtype="float32")
                y = paddle.to_tensor(np_y, dtype="float64")
                self.assertRaises(TypeError, paddle.divide, x=x, y=y)

                # rule 4: x is Tensor, y is scalar
                np_x = np.array([2, 3, 4])
                x = paddle.to_tensor(np_x, dtype="int32")
                y = 2
                z = x / y
                z_expected = np.array([1., 1.5, 2.])
                self.assertEqual((z_expected == z.numpy()).all(), True)

                # rule 5: y is Tensor, x is scalar
                np_x = np.array([2, 1, 4])
                x = paddle.to_tensor(np_x, dtype="int32")
                y = 2
                z = y / x
                z_expected = np.array([1., 2., 0.5])
                self.assertEqual((z_expected == z.numpy()).all(), True)

                # rule 6: y is Tensor, x is Tensor
                np_x = np.array([2, 3, 4])
                np_y = np.array([1, 5, 2])
                x = paddle.to_tensor(np_x)
                y = paddle.to_tensor(np_y)
                z = x / y
                z_expected = np.array([2., 0.6, 2.])
                self.assertEqual((z_expected == z.numpy()).all(), True)


if __name__ == '__main__':
    unittest.main()
