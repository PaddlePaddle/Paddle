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

import unittest

import numpy as np
from op_test import OpTest, convert_float_to_uint16, skip_check_grad_ci

import paddle
import paddle.fluid as fluid


class TestElementwiseOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype("float64"),
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'], 'Out', max_relative_error=0.005, no_grad_set=set("X")
        )

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'], 'Out', max_relative_error=0.005, no_grad_set=set('Y')
        )


class TestElementwiseSubOp_ZeroDim1(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, []).astype("float64"),
            'Y': np.random.uniform(0.1, 1, []).astype("float64"),
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}


class TestElementwiseSubOp_ZeroDim2(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype("float64"),
            'Y': np.random.uniform(0.1, 1, []).astype("float64"),
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}


class TestElementwiseSubOp_ZeroDim3(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.uniform(0.1, 1, []).astype("float64"),
            'Y': np.random.uniform(0.1, 1, [2, 3, 4, 5]).astype("float64"),
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}


class TestBF16ElementwiseOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.dtype = np.uint16
        x = np.random.uniform(0.1, 1, [13, 17]).astype(np.float32)
        y = np.random.uniform(0.1, 1, [13, 17]).astype(np.float32)
        out = x - y

        self.inputs = {
            'X': convert_float_to_uint16(x),
            'Y': convert_float_to_uint16(y),
        }
        self.outputs = {'Out': convert_float_to_uint16(out)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X', 'Y'], 'Out')

    def test_check_grad_ingore_x(self):
        self.check_grad(['Y'], 'Out', no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        self.check_grad(['X'], 'Out', no_grad_set=set("Y"))


class TestElementwiseFP16Op(TestElementwiseOp):
    def _executed_api(self, x, y, name=None):
        return paddle.subtract(x, y, name)

    def _executed_x_grad(self, x, y):
        paddle.disable_static()
        tensor_x = paddle.to_tensor(x)
        tensor_y = paddle.to_tensor(y)
        tensor_x.stop_gradient = False
        tensor_y.stop_gradient = False
        out = self._executed_api(tensor_x, tensor_y)
        grad = paddle.grad(out, [tensor_x], no_grad_vars=[tensor_y])
        x_grad = grad[0].numpy()
        paddle.enable_static()
        return x_grad

    def _executed_y_grad(self, x, y):
        paddle.disable_static()
        tensor_x = paddle.to_tensor(x)
        tensor_y = paddle.to_tensor(y)
        tensor_x.stop_gradient = False
        tensor_y.stop_gradient = False
        out = self._executed_api(tensor_x, tensor_y)
        grad = paddle.grad(out, [tensor_y], no_grad_vars=[tensor_x])
        y_grad = grad[0].numpy()
        paddle.enable_static()
        return y_grad

    def _executed_xy_grad(self, x, y):
        paddle.disable_static()
        tensor_x = paddle.to_tensor(x)
        tensor_y = paddle.to_tensor(y)
        tensor_x.stop_gradient = False
        tensor_y.stop_gradient = False
        out = self._executed_api(tensor_x, tensor_y)
        grad = paddle.grad(out, [tensor_x, tensor_y])
        paddle.enable_static()
        return [grad[0].numpy(), grad[1].numpy()]

    def _executed_ref_api(self, x, y, name=None):
        return x - y

    def _executed_ref_x_grad(self, x, y):
        return np.ones(x.shape).astype(np.float32)

    def _executed_ref_y_grad(self, x, y):
        return np.zeros(y.shape).astype(np.float32) - np.ones(y.shape).astype(
            np.float32
        )

    def _executed_ref_xy_grad(self, x, y):
        return [
            self._executed_ref_x_grad(x, y),
            self._executed_ref_y_grad(x, y),
        ]

    def test_check_output(self):
        ref_x = self.inputs['X'].astype(np.float32)
        ref_y = self.inputs['Y'].astype(np.float32)
        out_ref = self._executed_ref_api(ref_x, ref_y)
        paddle.disable_static()
        x = self.inputs['X'].astype(np.float16)
        y = self.inputs['Y'].astype(np.float16)
        tensor_x = paddle.to_tensor(x)
        tensor_y = paddle.to_tensor(y)
        out_pad = self._executed_api(tensor_x, tensor_y)
        paddle.enable_static()
        np.testing.assert_allclose(
            out_pad.numpy(), out_ref, rtol=1e-03, atol=1e-03
        )

    def test_check_grad(self):
        self.__class__.dtype = 'float16'
        ref_x = self.inputs['X'].astype(np.float32)
        ref_y = self.inputs['Y'].astype(np.float32)
        ref_xy_grad = self._executed_ref_xy_grad(ref_x, ref_y)
        x = self.inputs['X'].astype(np.float16)
        y = self.inputs['Y'].astype(np.float16)
        xy_grad = self._executed_xy_grad(x, y)
        np.testing.assert_allclose(
            ref_xy_grad[0], xy_grad[0], rtol=1e-03, atol=1e-03
        )
        np.testing.assert_allclose(
            ref_xy_grad[1], xy_grad[1], rtol=1e-03, atol=1e-03
        )

    def test_check_grad_ingore_y1(self):
        self.__class__.dtype = 'float16'
        ref_x = self.inputs['X'].astype(np.float32)
        ref_y = self.inputs['Y'].astype(np.float32)
        ref_x_grad = self._executed_ref_x_grad(ref_x, ref_y)
        x = self.inputs['X'].astype(np.float16)
        y = self.inputs['Y'].astype(np.float16)
        x_grad = self._executed_x_grad(x, y)
        np.testing.assert_allclose(ref_x_grad, x_grad, rtol=1e-03, atol=1e-03)

    def test_check_grad_ingore_x(self):
        self.__class__.dtype = 'float16'
        ref_x = self.inputs['X'].astype(np.float32)
        ref_y = self.inputs['Y'].astype(np.float32)
        ref_y_grad = self._executed_ref_y_grad(ref_x, ref_y)
        x = self.inputs['X'].astype(np.float16)
        y = self.inputs['Y'].astype(np.float16)
        y_grad = self._executed_y_grad(x, y)
        np.testing.assert_allclose(ref_y_grad, y_grad, rtol=1e-03, atol=1e-03)


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast."
)
class TestElementwiseSubOp_scalar(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(10, 3, 4).astype(np.float64),
            'Y': np.random.rand(1).astype(np.float64),
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}


class TestElementwiseSubOp_Vector(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.random((100,)).astype("float64"),
            'Y': np.random.random((100,)).astype("float64"),
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}


class TestElementwiseSubOp_broadcast_0(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(100, 3, 2).astype(np.float64),
            'Y': np.random.rand(100).astype(np.float64),
        }

        self.attrs = {'axis': 0}
        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(100, 1, 1)
        }


class TestElementwiseSubOp_broadcast_1(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(2, 100, 3).astype(np.float64),
            'Y': np.random.rand(100).astype(np.float64),
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 100, 1)
        }


class TestElementwiseSubOp_broadcast_2(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(2, 3, 100).astype(np.float64),
            'Y': np.random.rand(100).astype(np.float64),
        }

        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 1, 100)
        }


class TestElementwiseSubOp_broadcast_3(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(2, 10, 12, 3).astype(np.float64),
            'Y': np.random.rand(10, 12).astype(np.float64),
        }

        self.attrs = {'axis': 1}
        self.outputs = {
            'Out': self.inputs['X'] - self.inputs['Y'].reshape(1, 10, 12, 1)
        }


class TestElementwiseSubOp_broadcast_4(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(2, 5, 3, 12).astype(np.float64),
            'Y': np.random.rand(2, 5, 1, 12).astype(np.float64),
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}


class TestElementwiseSubOp_commonuse_1(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(2, 3, 100).astype(np.float64),
            'Y': np.random.rand(1, 1, 100).astype(np.float64),
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}


class TestElementwiseSubOp_commonuse_2(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(10, 3, 1, 4).astype(np.float64),
            'Y': np.random.rand(10, 1, 12, 1).astype(np.float64),
        }
        self.outputs = {'Out': self.inputs['X'] - self.inputs['Y']}


class TestElementwiseSubOp_xsize_lessthan_ysize(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.inputs = {
            'X': np.random.rand(10, 12).astype(np.float64),
            'Y': np.random.rand(2, 3, 10, 12).astype(np.float64),
        }

        self.attrs = {'axis': 2}

        self.outputs = {
            'Out': self.inputs['X'].reshape(1, 1, 10, 12) - self.inputs['Y']
        }


class TestComplexElementwiseSubOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_sub"
        self.dtype = np.float64
        self.shape = (2, 3, 4, 5)
        self.init_input_output()
        self.init_grad_input_output()

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(self.x),
            'Y': OpTest.np_dtype_to_fluid_dtype(self.y),
        }
        self.attrs = {'axis': -1, 'use_mkldnn': False}
        self.outputs = {'Out': self.out}

    def init_base_dtype(self):
        self.dtype = np.float64

    def init_input_output(self):
        self.x = np.random.random(self.shape).astype(
            self.dtype
        ) + 1j * np.random.random(self.shape).astype(self.dtype)
        self.y = np.random.random(self.shape).astype(
            self.dtype
        ) + 1j * np.random.random(self.shape).astype(self.dtype)
        self.out = self.x - self.y

    def init_grad_input_output(self):
        self.grad_out = np.ones(self.shape, self.dtype) + 1j * np.ones(
            self.shape, self.dtype
        )
        self.grad_x = self.grad_out
        self.grad_y = -self.grad_out

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            user_defined_grads=[self.grad_x, self.grad_y],
            user_defined_grad_outputs=[self.grad_out],
        )

    def test_check_grad_ingore_x(self):
        self.check_grad(
            ['Y'],
            'Out',
            no_grad_set=set("X"),
            user_defined_grads=[self.grad_y],
            user_defined_grad_outputs=[self.grad_out],
        )

    def test_check_grad_ingore_y(self):
        self.check_grad(
            ['X'],
            'Out',
            no_grad_set=set('Y'),
            user_defined_grads=[self.grad_x],
            user_defined_grad_outputs=[self.grad_out],
        )


class TestRealComplexElementwiseSubOp(TestComplexElementwiseSubOp):
    def init_input_output(self):
        self.x = np.random.random(self.shape).astype(self.dtype)
        self.y = np.random.random(self.shape).astype(
            self.dtype
        ) + 1j * np.random.random(self.shape).astype(self.dtype)
        self.out = self.x - self.y

    def init_grad_input_output(self):
        self.grad_out = np.ones(self.shape, self.dtype) + 1j * np.ones(
            self.shape, self.dtype
        )
        self.grad_x = np.real(self.grad_out)
        self.grad_y = -self.grad_out


class TestSubtractApi(unittest.TestCase):
    def _executed_api(self, x, y, name=None):
        return paddle.subtract(x, y, name)

    def test_name(self):
        with fluid.program_guard(fluid.Program()):
            x = fluid.data(name="x", shape=[2, 3], dtype="float32")
            y = fluid.data(name='y', shape=[2, 3], dtype='float32')

            y_1 = self._executed_api(x, y, name='subtract_res')
            self.assertEqual(('subtract_res' in y_1.name), True)

    def test_declarative(self):
        with fluid.program_guard(fluid.Program()):

            def gen_data():
                return {
                    "x": np.array([2, 3, 4]).astype('float32'),
                    "y": np.array([1, 5, 2]).astype('float32'),
                }

            x = fluid.data(name="x", shape=[3], dtype='float32')
            y = fluid.data(name="y", shape=[3], dtype='float32')
            z = self._executed_api(x, y)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            z_value = exe.run(feed=gen_data(), fetch_list=[z.name])
            z_expected = np.array([1.0, -2.0, 2.0])
            self.assertEqual((z_value == z_expected).all(), True)

    def test_dygraph(self):
        with fluid.dygraph.guard():
            np_x = np.array([2, 3, 4]).astype('float64')
            np_y = np.array([1, 5, 2]).astype('float64')
            x = fluid.dygraph.to_variable(np_x)
            y = fluid.dygraph.to_variable(np_y)
            z = self._executed_api(x, y)
            np_z = z.numpy()
            z_expected = np.array([1.0, -2.0, 2.0])
            self.assertEqual((np_z == z_expected).all(), True)


class TestSubtractInplaceApi(TestSubtractApi):
    def _executed_api(self, x, y, name=None):
        return x.subtract_(y, name)


class TestSubtractInplaceBroadcastSuccess(unittest.TestCase):
    def init_data(self):
        self.x_numpy = np.random.rand(2, 3, 4).astype('float')
        self.y_numpy = np.random.rand(3, 4).astype('float')

    def test_broadcast_success(self):
        paddle.disable_static()
        self.init_data()
        x = paddle.to_tensor(self.x_numpy)
        y = paddle.to_tensor(self.y_numpy)
        inplace_result = x.subtract_(y)
        numpy_result = self.x_numpy - self.y_numpy
        self.assertEqual((inplace_result.numpy() == numpy_result).all(), True)
        paddle.enable_static()


class TestSubtractInplaceBroadcastSuccess2(TestSubtractInplaceBroadcastSuccess):
    def init_data(self):
        self.x_numpy = np.random.rand(1, 2, 3, 1).astype('float')
        self.y_numpy = np.random.rand(3, 1).astype('float')


class TestSubtractInplaceBroadcastSuccess3(TestSubtractInplaceBroadcastSuccess):
    def init_data(self):
        self.x_numpy = np.random.rand(2, 3, 1, 5).astype('float')
        self.y_numpy = np.random.rand(1, 3, 1, 5).astype('float')


class TestSubtractInplaceBroadcastError(unittest.TestCase):
    def init_data(self):
        self.x_numpy = np.random.rand(3, 4).astype('float')
        self.y_numpy = np.random.rand(2, 3, 4).astype('float')

    def test_broadcast_errors(self):
        paddle.disable_static()
        self.init_data()
        x = paddle.to_tensor(self.x_numpy)
        y = paddle.to_tensor(self.y_numpy)

        def broadcast_shape_error():
            x.subtract_(y)

        self.assertRaises(ValueError, broadcast_shape_error)
        paddle.enable_static()


class TestSubtractInplaceBroadcastError2(TestSubtractInplaceBroadcastError):
    def init_data(self):
        self.x_numpy = np.random.rand(2, 1, 4).astype('float')
        self.y_numpy = np.random.rand(2, 3, 4).astype('float')


class TestSubtractInplaceBroadcastError3(TestSubtractInplaceBroadcastError):
    def init_data(self):
        self.x_numpy = np.random.rand(5, 2, 1, 4).astype('float')
        self.y_numpy = np.random.rand(2, 3, 4).astype('float')


class TestFloatElementwiseSubop(unittest.TestCase):
    def test_dygraph_sub(self):
        paddle.disable_static()

        np_a = np.random.random((2, 3, 4)).astype(np.float64)
        np_b = np.random.random((2, 3, 4)).astype(np.float64)

        tensor_a = paddle.to_tensor(np_a, dtype="float32")
        tensor_b = paddle.to_tensor(np_b, dtype="float32")

        # normal case: tensor - tensor
        expect_out = np_a - np_b
        actual_out = tensor_a - tensor_b
        np.testing.assert_allclose(
            actual_out, expect_out, rtol=1e-07, atol=1e-07
        )

        # normal case: tensor - scalar
        expect_out = np_a - 1
        actual_out = tensor_a - 1
        np.testing.assert_allclose(
            actual_out, expect_out, rtol=1e-07, atol=1e-07
        )

        # normal case: scalar - tenor
        expect_out = 1 - np_a
        actual_out = 1 - tensor_a
        np.testing.assert_allclose(
            actual_out, expect_out, rtol=1e-07, atol=1e-07
        )

        paddle.enable_static()


class TestFloatElementwiseSubop1(unittest.TestCase):
    def test_dygraph_sub(self):
        paddle.disable_static()

        np_a = np.random.random((2, 3, 4)).astype(np.float32)
        np_b = np.random.random((2, 3, 4)).astype(np.float32)

        tensor_a = paddle.to_tensor(np_a, dtype="float32")
        tensor_b = paddle.to_tensor(np_b, dtype="float32")

        # normal case: nparray - tenor
        expect_out = np_a - np_b
        actual_out = np_a - tensor_b
        np.testing.assert_allclose(
            actual_out, expect_out, rtol=1e-07, atol=1e-07
        )

        # normal case: tenor - nparray
        actual_out = tensor_a - np_b
        np.testing.assert_allclose(
            actual_out, expect_out, rtol=1e-07, atol=1e-07
        )

        paddle.enable_static()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
