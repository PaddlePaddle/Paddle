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
from op_test import (
    OpTest,
    check_cudnn_version_and_compute_capability,
    convert_float_to_uint16,
    skip_check_grad_ci,
)

import paddle

paddle.enable_static()


def broadcast_wrapper(shape=[1, 10, 12, 1]):
    def min_wrapper(x, y, axis=-1):
        return paddle.minimum(x, y.reshape(shape))

    return min_wrapper


class TestElementwiseOp(OpTest):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        self.public_python_api = paddle.minimum
        self.prim_op_type = "prim"
        self.if_enable_cinn()
        # If x and y have the same value, the min() is not differentiable.
        # So we generate test data by the following method
        # to avoid them being too close to each other.
        x = np.random.uniform(0.1, 1, [13, 17]).astype("float64")
        sgn = np.random.choice([-1, 1], [13, 17]).astype("float64")
        y = x + sgn * np.random.uniform(0.1, 1, [13, 17]).astype("float64")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        if hasattr(self, 'attrs'):
            if self.attrs['axis'] == -1:
                self.check_grad(
                    ['X', 'Y'], 'Out', check_prim=False, check_prim_pir=True
                )
            else:
                self.check_grad(['X', 'Y'], 'Out')
        else:
            self.check_grad(
                ['X', 'Y'], 'Out', check_prim=False, check_prim_pir=True
            )

    def test_check_grad_ignore_x(self):
        if hasattr(self, 'attrs') and self.attrs['axis'] != -1:
            self.check_grad(
                ['Y'],
                'Out',
                max_relative_error=0.005,
                no_grad_set=set("X"),
            )
        else:
            self.check_grad(
                ['Y'],
                'Out',
                max_relative_error=0.005,
                no_grad_set=set("X"),
                check_prim=False,
                check_prim_pir=True,
            )

    def test_check_grad_ignore_y(self):
        if hasattr(self, 'attrs') and self.attrs['axis'] != -1:
            self.check_grad(
                ['X'],
                'Out',
                max_relative_error=0.005,
                no_grad_set=set('Y'),
                check_dygraph=False,
            )
        else:
            self.check_grad(
                ['X'],
                'Out',
                max_relative_error=0.005,
                no_grad_set=set('Y'),
                check_prim=False,
                check_prim_pir=True,
            )

    def if_enable_cinn(self):
        pass


class TestElementwiseFP16Op(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        self.public_python_api = paddle.minimum
        self.prim_op_type = "prim"
        self.if_enable_cinn()
        self.dtype = np.float16
        # If x and y have the same value, the min() is not differentiable.
        # So we generate test data by the following method
        # to avoid them being too close to each other.
        x = np.random.uniform(0.1, 1, [13, 17]).astype(np.float16)
        sgn = np.random.choice([-1, 1], [13, 17]).astype(np.float16)
        y = x + sgn * np.random.uniform(0.1, 1, [13, 17]).astype(np.float16)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMinOp_ZeroDim1(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        self.public_python_api = paddle.minimum
        self.prim_op_type = "prim"
        self.if_enable_cinn()
        x = np.random.uniform(0.1, 1, []).astype("float64")
        y = np.random.uniform(0.1, 1, []).astype("float64")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMinFP16Op_ZeroDim1(TestElementwiseFP16Op):
    def init_data(self):
        self.x = np.random.uniform(0.1, 1, []).astype(np.float16)
        self.y = np.random.uniform(0.1, 1, []).astype(np.float16)


class TestElementwiseMinOp_ZeroDim2(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        self.public_python_api = paddle.minimum
        self.prim_op_type = "prim"
        x = np.random.uniform(0.1, 1, [13, 17]).astype("float64")
        y = np.random.uniform(0.1, 1, []).astype("float64")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMinFP16Op_ZeroDim2(TestElementwiseFP16Op):
    def init_data(self):
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype("float16")
        self.y = np.random.uniform(0.1, 1, []).astype("float16")


class TestElementwiseMinOp_ZeroDim3(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        self.public_python_api = paddle.minimum
        self.prim_op_type = "prim"
        self.if_enable_cinn()
        x = np.random.uniform(0.1, 1, []).astype("float64")
        y = np.random.uniform(0.1, 1, [13, 17]).astype("float64")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMinFP16Op_ZeroDim3(TestElementwiseFP16Op):
    def init_data(self):
        self.x = np.random.uniform(0.1, 1, []).astype("float16")
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype("float16")


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast."
)
class TestElementwiseMinOp_scalar(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        self.public_python_api = paddle.minimum
        self.prim_op_type = "prim"
        self.if_enable_cinn()
        x = np.random.random_integers(-5, 5, [10, 3, 4]).astype("float64")
        y = np.array([0.5]).astype("float64")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}


@skip_check_grad_ci(
    reason="[skip shape check] Use y_shape(1) to test broadcast."
)
class TestElementwiseMinFP16Op_scalar(TestElementwiseFP16Op):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        self.public_python_api = paddle.minimum
        self.prim_op_type = "prim"
        self.if_enable_cinn()
        x = np.random.random_integers(-5, 5, [10, 3, 4]).astype(np.float16)
        y = np.array([0.5]).astype(np.float16)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMinOp_Vector(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        self.public_python_api = paddle.minimum
        self.prim_op_type = "prim"
        self.if_enable_cinn()
        x = np.random.random((100,)).astype("float64")
        sgn = np.random.choice([-1, 1], (100,)).astype("float64")
        y = x + sgn * np.random.uniform(0.1, 1, (100,)).astype("float64")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMinFP16Op_Vector(TestElementwiseFP16Op):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        self.public_python_api = paddle.minimum
        self.prim_op_type = "prim"
        self.if_enable_cinn()
        x = np.random.random((100,)).astype(np.float16)
        sgn = np.random.choice([-1, 1], (100,)).astype(np.float16)
        y = x + sgn * np.random.uniform(0.1, 1, (100,)).astype(np.float16)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMinOp_broadcast_2(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = broadcast_wrapper(shape=[1, 1, 100])
        self.public_python_api = paddle.minimum
        self.prim_op_type = "prim"
        self.if_enable_cinn()
        x = np.random.uniform(0.5, 1, (2, 3, 100)).astype(np.float64)
        sgn = np.random.choice([-1, 1], (100,)).astype(np.float64)
        y = x[0, 0, :] + sgn * np.random.uniform(1, 2, (100,)).astype(
            np.float64
        )
        self.inputs = {'X': x, 'Y': y}

        self.outputs = {
            'Out': np.minimum(
                self.inputs['X'], self.inputs['Y'].reshape(1, 1, 100)
            )
        }


class TestElementwiseMinFP16Op_broadcast_2(TestElementwiseFP16Op):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = broadcast_wrapper(shape=[1, 1, 100])
        self.public_python_api = paddle.minimum
        self.prim_op_type = "prim"
        self.if_enable_cinn()
        x = np.random.uniform(0.5, 1, (2, 3, 100)).astype(np.float16)
        sgn = np.random.choice([-1, 1], (100,)).astype(np.float16)
        y = x[0, 0, :] + sgn * np.random.uniform(1, 2, (100,)).astype(
            np.float16
        )
        self.inputs = {'X': x, 'Y': y}

        self.outputs = {
            'Out': np.minimum(
                self.inputs['X'], self.inputs['Y'].reshape(1, 1, 100)
            )
        }


class TestElementwiseMinOp_broadcast_4(TestElementwiseOp):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        self.prim_op_type = "prim"
        self.public_python_api = paddle.minimum
        self.if_enable_cinn()
        x = np.random.uniform(0.5, 1, (2, 10, 2, 5)).astype(np.float64)
        sgn = np.random.choice([-1, 1], (2, 10, 1, 5)).astype(np.float64)
        y = x + sgn * np.random.uniform(1, 2, (2, 10, 1, 5)).astype(np.float64)
        self.inputs = {'X': x, 'Y': y}

        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseMinFP16Op_broadcast_4(TestElementwiseFP16Op):
    def setUp(self):
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        self.public_python_api = paddle.minimum
        self.prim_op_type = "prim"
        self.if_enable_cinn()
        x = np.random.uniform(0.5, 1, (2, 10, 2, 5)).astype(np.float16)
        sgn = np.random.choice([-1, 1], (2, 10, 1, 5)).astype(np.float16)
        y = x + sgn * np.random.uniform(1, 2, (2, 10, 1, 5)).astype(np.float16)
        self.inputs = {'X': x, 'Y': y}

        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}


@unittest.skipIf(
    not check_cudnn_version_and_compute_capability(8100, 8),
    "only support compiled with CUDA or custom device, and for CUDA cudnn version need larger than 8.1.0 and device's compute capability is at least 8.0",
)
class TestElementwiseBF16Op(OpTest):
    def init_data(self):
        # If x and y have the same value, the max() is not differentiable.
        # So we generate test data by the following method
        # to avoid them being too close to each other.
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(np.float32)
        sgn = np.random.choice([-1, 1], [13, 17]).astype(np.float32)
        self.y = self.x + sgn * np.random.uniform(0.1, 1, [13, 17]).astype(
            np.float32
        )

    def setUp(self):
        self.init_data()
        self.op_type = "elementwise_min"
        self.python_api = paddle.minimum
        self.public_python_api = paddle.minimum
        self.prim_op_type = "prim"
        self.if_enable_cinn()
        self.dtype = np.uint16
        self.inputs = {
            'X': convert_float_to_uint16(self.x),
            'Y': convert_float_to_uint16(self.y),
        }
        self.outputs = {
            'Out': convert_float_to_uint16(np.minimum(self.x, self.y))
        }

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        places = self._get_places()
        for place in places:
            if type(place) is paddle.base.libpaddle.CPUPlace:
                check_prim = False
            else:
                check_prim = True

            self.check_grad_with_place(
                place,
                inputs_to_check=['X', 'Y'],
                output_names='Out',
                no_grad_set=None,
                numeric_grad_delta=0.05,
                in_place=False,
                max_relative_error=0.005,
                user_defined_grads=None,
                user_defined_grad_outputs=None,
                check_dygraph=True,
                check_prim=False,
                only_check_prim=False,
                atol=1e-5,
                check_cinn=False,
                check_prim_pir=check_prim,
            )

    def test_check_grad_ignore_x(self):
        places = self._get_places()
        for place in places:
            if isinstance(place, paddle.base.libpaddle.CPUPlace):
                check_prim = False
            else:
                check_prim = True

            self.check_grad_with_place(
                place,
                inputs_to_check=['Y'],
                output_names='Out',
                no_grad_set=set("X"),
                numeric_grad_delta=0.05,
                in_place=False,
                max_relative_error=0.005,
                user_defined_grads=None,
                user_defined_grad_outputs=None,
                check_dygraph=True,
                check_prim=False,
                only_check_prim=False,
                atol=1e-5,
                check_cinn=False,
                check_prim_pir=check_prim,
            )

    def test_check_grad_ignore_y(self):
        places = self._get_places()
        for place in places:
            if isinstance(place, paddle.base.libpaddle.CPUPlace):
                check_prim = False
            else:
                check_prim = True

            self.check_grad_with_place(
                place,
                inputs_to_check=['Y'],
                output_names='Out',
                no_grad_set=set("X"),
                numeric_grad_delta=0.05,
                in_place=False,
                max_relative_error=0.005,
                user_defined_grads=None,
                user_defined_grad_outputs=None,
                check_dygraph=True,
                check_prim=False,
                only_check_prim=False,
                atol=1e-5,
                check_cinn=False,
                check_prim_pir=check_prim,
            )

    def if_enable_cinn(self):
        pass


class TestElementwiseMinBF16Op_ZeroDim1(TestElementwiseBF16Op):
    def init_data(self):
        self.x = np.random.uniform(0.1, 1, []).astype("float32")
        self.y = np.random.uniform(0.1, 1, []).astype("float32")


class TestElementwiseMinBF16Op_scalar(TestElementwiseBF16Op):
    def init_data(self):
        self.x = np.random.random_integers(-5, 5, [2, 3, 20]).astype("float32")
        self.y = np.array([0.5]).astype("float32")
        self.__class__.no_need_check_grad = True


class TestElementwiseMinBF16Op_Vector(TestElementwiseBF16Op):
    def init_data(self):
        self.x = np.random.random((100,)).astype("float32")
        sgn = np.random.choice([-1, 1], (100,)).astype("float32")
        self.y = self.x + sgn * np.random.uniform(0.1, 1, (100,)).astype(
            "float32"
        )


class TestElementwiseOpEqualInput(TestElementwiseOp):
    def init_data(self):
        self.x = np.ones([13, 17]).astype(np.float32)
        self.y = np.ones([13, 17]).astype(np.float32)

    def setUp(self):
        self.init_data()
        self.op_type = "elementwise_min"
        self.prim_op_type = "prim"
        self.if_enable_cinn()
        self.python_api = paddle.minimum
        self.dtype = np.float32
        self.public_python_api = paddle.minimum
        self.inputs = {'X': self.x, 'Y': self.y}
        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}


class TestElementwiseOp0SizeInput(TestElementwiseOp):
    def init_data(self):
        self.x = np.ones([0, 1, 2]).astype(np.float32)
        self.y = np.ones([1, 3598, 2]).astype(np.float32)

    def setUp(self):
        self.init_data()
        self.op_type = "elementwise_min"
        self.prim_op_type = "prim"
        self.if_enable_cinn()
        self.python_api = paddle.minimum
        self.dtype = np.float32
        self.public_python_api = paddle.minimum
        self.inputs = {'X': self.x, 'Y': self.y}
        self.outputs = {'Out': np.minimum(self.inputs['X'], self.inputs['Y'])}


class TestMinimumOutAndAlias(unittest.TestCase):
    def test_dygraph(self):
        paddle.disable_static()
        x = paddle.to_tensor(
            np.array([[1, 2], [7, 8]]), dtype='float32', stop_gradient=False
        )
        y = paddle.to_tensor(
            np.array([[3, 4], [5, 6]]), dtype='float32', stop_gradient=False
        )

        def run_case(case):
            out_buf = paddle.zeros_like(x)
            out_buf.stop_gradient = False
            if case == 'return':
                z = paddle.minimum(x, y)
            elif case == 'input_out':
                paddle.minimum(x, y, out=out_buf)
                z = out_buf
            elif case == 'both_return':
                z = paddle.minimum(input=x, other=y, out=out_buf)
            elif case == 'both_input_out':
                _ = paddle.minimum(input=x, other=y, out=out_buf)
                z = out_buf
            else:
                raise AssertionError
            ref = paddle._C_ops.minimum(x, y)
            np.testing.assert_allclose(
                z.numpy(), ref.numpy(), rtol=1e-6, atol=1e-6
            )
            (z.mean()).backward()
            return z.numpy(), x.grad.numpy(), y.grad.numpy()

        z1, gx1, gy1 = run_case('return')
        x.clear_gradient()
        y.clear_gradient()
        z2, gx2, gy2 = run_case('input_out')
        x.clear_gradient()
        y.clear_gradient()
        z3, gx3, gy3 = run_case('both_return')
        x.clear_gradient()
        y.clear_gradient()
        z4, gx4, gy4 = run_case('both_input_out')

        np.testing.assert_allclose(z1, z2, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(z1, z3, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(z1, z4, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(gx1, gx2, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(gx1, gx3, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(gx1, gx4, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(gy1, gy2, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(gy1, gy3, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(gy1, gy4, rtol=1e-6, atol=1e-6)

        paddle.enable_static()

    def test_static(self):
        paddle.enable_static()
        startup_prog = paddle.static.Program()
        main_prog = paddle.static.Program()

        with paddle.static.program_guard(main_prog, startup_prog):
            x = paddle.static.data('X', [5, 7], 'float32')
            y = paddle.static.data('Y', [5, 7], 'float32')
            z = paddle.minimum(input=x, other=y)

        x_data = np.random.random([5, 7]).astype('float32')
        y_data = np.random.random([5, 7]).astype('float32')
        ref = np.minimum(x_data, y_data)

        exe = paddle.static.Executor(paddle.CPUPlace())
        exe.run(startup_prog)
        out = exe.run(
            main_prog,
            feed={'X': x_data, 'Y': y_data},
            fetch_list=[z],
        )
        np.testing.assert_allclose(out[0], ref, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
