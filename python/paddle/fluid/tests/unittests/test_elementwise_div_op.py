#  Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
from paddle import fluid
from paddle.fluid import core


class ElementwiseDivOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.python_api = paddle.divide
        self.init_args()
        self.init_dtype()
        self.init_shape()

        x = self.gen_data(self.x_shape).astype(self.val_dtype)
        y = self.gen_data(self.y_shape).astype(self.val_dtype)
        out = self.compute_output(x, y).astype(self.val_dtype)
        grad_out = np.ones(out.shape).astype(self.val_dtype)
        grad_x = self.compute_gradient_x(grad_out, y).astype(self.val_dtype)
        grad_y = self.compute_gradient_y(grad_out, out, y).astype(
            self.val_dtype
        )

        # Convert np.float32 data to np.uint16 for bfloat16 Paddle OP
        if self.dtype == np.uint16:
            x = convert_float_to_uint16(x)
            y = convert_float_to_uint16(y)
            out = convert_float_to_uint16(out)
            grad_out = convert_float_to_uint16(grad_out)
            grad_x = convert_float_to_uint16(grad_x)
            grad_y = convert_float_to_uint16(grad_y)

        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out}
        self.grad_out = grad_out
        self.grad_x = grad_x
        self.grad_y = grad_y

    def init_args(self):
        self.check_dygraph = True
        self.place = None

    def init_dtype(self):
        self.dtype = np.float64
        self.val_dtype = np.float64

    def init_shape(self):
        self.x_shape = [13, 17]
        self.y_shape = [13, 17]

    def gen_data(self, shape):
        return np.random.uniform(0.1, 1, shape)

    def compute_output(self, x, y):
        return x / y

    def compute_gradient_x(self, grad_out, y):
        return grad_out / y

    def compute_gradient_y(self, grad_out, out, y):
        return -1 * grad_out * out / y

    def test_check_output(self):
        if self.place is None:
            self.check_output()
        else:
            self.check_output_with_place(self.place)

    def test_check_gradient(self):
        check_list = []
        check_list.append(
            {
                'grad': ['X', 'Y'],
                'no_grad': None,
                'val_grad': [self.grad_x, self.grad_y],
            }
        )
        check_list.append(
            {'grad': ['Y'], 'no_grad': set('X'), 'val_grad': [self.grad_y]}
        )
        check_list.append(
            {'grad': ['X'], 'no_grad': set('Y'), 'val_grad': [self.grad_x]}
        )
        for check_option in check_list:
            check_args = [check_option['grad'], 'Out']
            check_kwargs = {
                'no_grad_set': check_option['no_grad'],
                'user_defined_grads': check_option['val_grad'],
                'user_defined_grad_outputs': [self.grad_out],
                'check_dygraph': self.check_dygraph,
            }
            if self.place is None:
                self.check_grad(*check_args, **check_kwargs)
            else:
                check_args.insert(0, self.place)
                self.check_grad_with_place(*check_args, **check_kwargs)


class TestElementwiseDivOp_ZeroDim1(ElementwiseDivOp):
    def init_shape(self):
        self.x_shape = []
        self.y_shape = []


class TestElementwiseDivOp_ZeroDim2(ElementwiseDivOp):
    def init_shape(self):
        self.x_shape = [13, 17]
        self.y_shape = []

    def compute_output(self, x, y):
        return x / y.reshape([1, 1])

    def compute_gradient_x(self, grad_out, y):
        return grad_out / y.reshape([1, 1])

    def compute_gradient_y(self, grad_out, out, y):
        return np.sum(-1 * grad_out * out / y.reshape([1, 1]))


class TestElementwiseDivOp_ZeroDim3(ElementwiseDivOp):
    def init_shape(self):
        self.x_shape = []
        self.y_shape = [13, 17]

    def compute_output(self, x, y):
        return x.reshape([1, 1]) / y

    def compute_gradient_x(self, grad_out, y):
        return np.sum(grad_out / y)

    def compute_gradient_y(self, grad_out, out, y):
        return -1 * grad_out * out / y


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestElementwiseDivOpBF16(ElementwiseDivOp):
    def init_args(self):
        # In due to output data type inconsistence of bfloat16 paddle op, we disable the dygraph check.
        self.check_dygraph = False
        self.place = core.CUDAPlace(0)

    def init_dtype(self):
        self.dtype = np.uint16
        self.val_dtype = np.float32

    def init_shape(self):
        self.x_shape = [12, 13]
        self.y_shape = [12, 13]


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast."
)
class TestElementwiseDivOpScalar(ElementwiseDivOp):
    def init_shape(self):
        self.x_shape = [20, 3, 4]
        self.y_shape = [1]

    def compute_gradient_y(self, grad_out, out, y):
        return np.array([np.sum(-1 * grad_out * out / y)])


class TestElementwiseDivOpVector(ElementwiseDivOp):
    def init_shape(self):
        self.x_shape = [100]
        self.y_shape = [100]


class TestElementwiseDivOpBroadcast0(ElementwiseDivOp):
    def init_shape(self):
        self.x_shape = [100, 3, 4]
        self.y_shape = [100]
        self.attrs = {'axis': 0}

    def compute_output(self, x, y):
        return x / y.reshape(100, 1, 1)

    def compute_gradient_x(self, grad_out, y):
        return grad_out / y.reshape(100, 1, 1)

    def compute_gradient_y(self, grad_out, out, y):
        return np.sum(-1 * grad_out * out / y.reshape(100, 1, 1), axis=(1, 2))


class TestElementwiseDivOpBroadcast1(ElementwiseDivOp):
    def init_shape(self):
        self.x_shape = [2, 100, 4]
        self.y_shape = [100]
        self.attrs = {'axis': 1}

    def compute_output(self, x, y):
        return x / y.reshape(1, 100, 1)

    def compute_gradient_x(self, grad_out, y):
        return grad_out / y.reshape(1, 100, 1)

    def compute_gradient_y(self, grad_out, out, y):
        return np.sum(-1 * grad_out * out / y.reshape(1, 100, 1), axis=(0, 2))


class TestElementwiseDivOpBroadcast2(ElementwiseDivOp):
    def init_shape(self):
        self.x_shape = [2, 3, 100]
        self.y_shape = [100]

    def compute_output(self, x, y):
        return x / y.reshape(1, 1, 100)

    def compute_gradient_x(self, grad_out, y):
        return grad_out / y.reshape(1, 1, 100)

    def compute_gradient_y(self, grad_out, out, y):
        return np.sum(-1 * grad_out * out / y.reshape(1, 1, 100), axis=(0, 1))


class TestElementwiseDivOpBroadcast3(ElementwiseDivOp):
    def init_shape(self):
        self.x_shape = [2, 10, 12, 5]
        self.y_shape = [10, 12]
        self.attrs = {'axis': 1}

    def compute_output(self, x, y):
        return x / y.reshape(1, 10, 12, 1)

    def compute_gradient_x(self, grad_out, y):
        return grad_out / y.reshape(1, 10, 12, 1)

    def compute_gradient_y(self, grad_out, out, y):
        return np.sum(
            -1 * grad_out * out / y.reshape(1, 10, 12, 1), axis=(0, 3)
        )


class TestElementwiseDivOpBroadcast4(ElementwiseDivOp):
    def init_shape(self):
        self.x_shape = [2, 3, 50]
        self.y_shape = [2, 1, 50]

    def compute_gradient_y(self, grad_out, out, y):
        return np.sum(-1 * grad_out * out / y, axis=(1)).reshape(2, 1, 50)


class TestElementwiseDivOpBroadcast5(ElementwiseDivOp):
    def init_shape(self):
        self.x_shape = [2, 3, 4, 20]
        self.y_shape = [2, 3, 1, 20]

    def compute_gradient_y(self, grad_out, out, y):
        return np.sum(-1 * grad_out * out / y, axis=(2)).reshape(2, 3, 1, 20)


class TestElementwiseDivOpCommonuse1(ElementwiseDivOp):
    def init_shape(self):
        self.x_shape = [2, 3, 100]
        self.y_shape = [1, 1, 100]

    def compute_gradient_y(self, grad_out, out, y):
        return np.sum(-1 * grad_out * out / y, axis=(0, 1)).reshape(1, 1, 100)


class TestElementwiseDivOpCommonuse2(ElementwiseDivOp):
    def init_shape(self):
        self.x_shape = [30, 3, 1, 5]
        self.y_shape = [30, 1, 4, 1]

    def compute_gradient_x(self, grad_out, y):
        return np.sum(grad_out / y, axis=(2)).reshape(30, 3, 1, 5)

    def compute_gradient_y(self, grad_out, out, y):
        return np.sum(-1 * grad_out * out / y, axis=(1, 3)).reshape(30, 1, 4, 1)


class TestElementwiseDivOpXsizeLessThanYsize(ElementwiseDivOp):
    def init_shape(self):
        self.x_shape = [10, 12]
        self.y_shape = [2, 3, 10, 12]
        self.attrs = {'axis': 2}

    def compute_gradient_x(self, grad_out, y):
        return np.sum(grad_out / y, axis=(0, 1))


class TestElementwiseDivOpInt(ElementwiseDivOp):
    def init_dtype(self):
        self.dtype = np.int32
        self.val_dtype = np.int32

    def gen_data(self, shape):
        return np.random.randint(1, 5, size=shape)

    def compute_output(self, x, y):
        return x // y


@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestElementwiseDivOpFp16(ElementwiseDivOp):
    def init_dtype(self):
        self.dtype = np.float16
        self.val_dtype = np.float16


class TestElementwiseDivBroadcast(unittest.TestCase):
    def test_shape_with_batch_sizes(self):
        with fluid.program_guard(fluid.Program()):
            x_var = fluid.data(
                name='x', dtype='float32', shape=[None, 3, None, None]
            )
            one = 2.0
            out = one / x_var
            exe = fluid.Executor(fluid.CPUPlace())
            x = np.random.uniform(0.1, 0.6, (1, 3, 32, 32)).astype("float32")
            (out_result,) = exe.run(feed={'x': x}, fetch_list=[out])
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
            z_expected = np.array([2.0, 0.6, 2.0])
            self.assertEqual((np_z == z_expected).all(), True)


class TestComplexElementwiseDivOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_div"
        self.python_api = paddle.divide
        self.init_base_dtype()
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
        self.x = np.random.random((2, 3, 4, 5)).astype(
            self.dtype
        ) + 1j * np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.y = np.random.random((2, 3, 4, 5)).astype(
            self.dtype
        ) + 1j * np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.out = self.x / self.y

    def init_grad_input_output(self):
        self.grad_out = np.ones((2, 3, 4, 5), self.dtype) + 1j * np.ones(
            (2, 3, 4, 5), self.dtype
        )
        self.grad_x = self.grad_out / np.conj(self.y)
        self.grad_y = -self.grad_out * np.conj(self.x / self.y / self.y)

    def test_check_output(self):
        self.check_output(check_eager=False)

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


class TestRealComplexElementwiseDivOp(TestComplexElementwiseDivOp):
    def init_input_output(self):
        self.x = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.y = np.random.random((2, 3, 4, 5)).astype(
            self.dtype
        ) + 1j * np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.out = self.x / self.y

    def init_grad_input_output(self):
        self.grad_out = np.ones((2, 3, 4, 5), self.dtype) + 1j * np.ones(
            (2, 3, 4, 5), self.dtype
        )
        self.grad_x = np.real(self.grad_out / np.conj(self.y))
        self.grad_y = -self.grad_out * np.conj(self.x / self.y / self.y)


class TestElementwiseDivop(unittest.TestCase):
    def test_dygraph_div(self):
        paddle.disable_static()

        np_a = np.random.random((2, 3, 4)).astype(np.float32)
        np_b = np.random.random((2, 3, 4)).astype(np.float32)
        np_a[np.abs(np_a) < 0.0005] = 0.002
        np_b[np.abs(np_b) < 0.0005] = 0.002

        tensor_a = paddle.to_tensor(np_a, dtype="float32")
        tensor_b = paddle.to_tensor(np_b, dtype="float32")

        # normal case: nparray / tenor
        expect_out = np_a / np_b
        actual_out = np_a / tensor_b
        np.testing.assert_allclose(actual_out, expect_out)

        # normal case: tensor / nparray
        actual_out = tensor_a / np_b
        np.testing.assert_allclose(actual_out, expect_out)

        paddle.enable_static()


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
