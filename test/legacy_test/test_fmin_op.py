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

import unittest

import numpy as np
from op_test import (
    OpTest,
    convert_float_to_uint16,
    get_device_place,
    is_custom_device,
)

import paddle
from paddle.base import core

paddle.enable_static()


class ApiFMinTest(unittest.TestCase):
    """ApiFMinTest"""

    def setUp(self):
        """setUp"""
        if core.is_compiled_with_cuda() or is_custom_device():
            self.place = get_device_place()
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
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            data_x = paddle.static.data("x", shape=[10, 15], dtype="float32")
            data_y = paddle.static.data("y", shape=[10, 15], dtype="float32")
            result_fmin = paddle.fmin(data_x, data_y)
            exe = paddle.static.Executor(self.place)
            (res,) = exe.run(
                feed={"x": self.input_x, "y": self.input_y},
                fetch_list=[result_fmin],
            )
        np.testing.assert_allclose(res, self.np_expected1, rtol=1e-05)

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            data_x = paddle.static.data("x", shape=[10, 15], dtype="float32")
            data_z = paddle.static.data("z", shape=[15], dtype="float32")
            result_fmin = paddle.fmin(data_x, data_z)
            exe = paddle.static.Executor(self.place)
            (res,) = exe.run(
                feed={"x": self.input_x, "z": self.input_z},
                fetch_list=[result_fmin],
            )
        np.testing.assert_allclose(res, self.np_expected2, rtol=1e-05)

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            data_a = paddle.static.data("a", shape=[3], dtype="int64")
            data_c = paddle.static.data("c", shape=[3], dtype="int64")
            result_fmin = paddle.fmin(data_a, data_c)
            exe = paddle.static.Executor(self.place)
            (res,) = exe.run(
                feed={"a": self.input_a, "c": self.input_c},
                fetch_list=[result_fmin],
            )
        np.testing.assert_allclose(res, self.np_expected3, rtol=1e-05)

        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            data_b = paddle.static.data("b", shape=[3], dtype="int64")
            data_c = paddle.static.data("c", shape=[3], dtype="int64")
            result_fmin = paddle.fmin(data_b, data_c)
            exe = paddle.static.Executor(self.place)
            (res,) = exe.run(
                feed={"b": self.input_b, "c": self.input_c},
                fetch_list=[result_fmin],
            )
        np.testing.assert_allclose(res, self.np_expected4, rtol=1e-05)

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
        np.testing.assert_allclose(res, self.np_expected1, rtol=1e-05)

        # test broadcast
        res = paddle.fmin(x, z)
        res = res.numpy()
        np.testing.assert_allclose(res, self.np_expected2, rtol=1e-05)

        res = paddle.fmin(a, c)
        res = res.numpy()
        np.testing.assert_allclose(res, self.np_expected3, rtol=1e-05)

        res = paddle.fmin(b, c)
        res = res.numpy()
        np.testing.assert_allclose(res, self.np_expected4, rtol=1e-05)


class TestElementwiseFminOp(OpTest):
    """TestElementwiseFminOp"""

    def setUp(self):
        """setUp"""
        self.op_type = "elementwise_fmin"
        self.prim_op_type = "prim"
        self.python_api = paddle.fmin
        self.public_python_api = paddle.fmin
        # If x and y have the same value, the min() is not differentiable.
        # So we generate test data by the following method
        # to avoid them being too close to each other.
        self.init_shape()
        x = np.random.uniform(0.1, 1, self.shape).astype("float64")
        sgn = np.random.choice([-1, 1], self.shape).astype("float64")
        y = x + sgn * np.random.uniform(0.1, 1, self.shape).astype("float64")
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.fmin(self.inputs['X'], self.inputs['Y'])}

    def init_shape(self):
        """init_shape"""
        self.shape = [13, 17]

    def test_check_output(self):
        """test_check_output"""
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad_normal(self):
        """test_check_grad_normal"""
        self.check_grad(['X', 'Y'], 'Out', check_pir=True, check_prim_pir=True)

    def test_check_grad_ignore_x(self):
        """test_check_grad_ignore_x"""
        self.check_grad(
            ['Y'],
            'Out',
            max_relative_error=0.005,
            no_grad_set=set("X"),
            check_pir=True,
        )

    def test_check_grad_ignore_y(self):
        """test_check_grad_ignore_y"""
        self.check_grad(
            ['X'],
            'Out',
            max_relative_error=0.005,
            no_grad_set=set('Y'),
            check_pir=True,
        )


class TestElementwiseFmin2Op(OpTest):
    """TestElementwiseFmin2Op"""

    def setUp(self):
        """setUp"""
        self.op_type = "elementwise_fmin"
        self.prim_op_type = "prim"
        self.python_api = paddle.fmin
        self.public_python_api = paddle.fmin
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
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad_normal(self):
        """test_check_grad_normal"""
        self.check_grad(['X', 'Y'], 'Out', check_pir=True, check_prim_pir=True)

    def test_check_grad_ignore_x(self):
        """test_check_grad_ignore_x"""
        self.check_grad(
            ['Y'],
            'Out',
            max_relative_error=0.005,
            no_grad_set=set("X"),
            check_pir=True,
        )

    def test_check_grad_ignore_y(self):
        """test_check_grad_ignore_y"""
        self.check_grad(
            ['X'],
            'Out',
            max_relative_error=0.005,
            no_grad_set=set('Y'),
            check_pir=True,
        )


class TestElementwiseFmin3Op(OpTest):
    """TestElementwiseFmin2Op"""

    def setUp(self):
        """setUp"""
        self.op_type = "elementwise_fmin"
        self.prim_op_type = "prim"
        self.python_api = paddle.fmin
        self.public_python_api = paddle.fmin
        # If x and y have the same value, the min() is not differentiable.
        # So we generate test data by the following method
        # to avoid them being too close to each other.
        x = np.random.uniform(1, 1, [13, 17]).astype("float16")
        sgn = np.random.choice([-1, 1], [13, 17]).astype("float16")
        y = x + sgn * np.random.uniform(1, 1, [13, 17]).astype("float16")

        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': np.fmin(self.inputs['X'], self.inputs['Y'])}

    def test_check_output(self):
        """test_check_output"""
        self.check_output(check_pir=True, check_symbol_infer=False)

    def test_check_grad_normal(self):
        """test_check_grad_normal"""
        self.check_grad(['X', 'Y'], 'Out', check_pir=True, check_prim_pir=True)


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device())
    or not core.is_bfloat16_supported(get_device_place()),
    "core is not compiled with CUDA and not support the bfloat16",
)
class TestFminBF16OP(OpTest):
    def setUp(self):
        self.op_type = "elementwise_fmin"
        self.prim_op_type = "prim"
        self.python_api = paddle.fmin
        self.public_python_api = paddle.fmin
        self.dtype = np.uint16
        x = np.random.uniform(1, 1, [13, 17]).astype("float32")
        sgn = np.random.choice([-1, 1], [13, 17]).astype("float32")
        y = x + sgn * np.random.uniform(1, 1, [13, 17]).astype("float32")
        out = np.fmin(x, y)
        self.inputs = {
            'X': convert_float_to_uint16(x),
            'Y': convert_float_to_uint16(y),
        }
        self.outputs = {'Out': convert_float_to_uint16(out)}

    def test_check_output(self):
        place = get_device_place()
        self.check_output_with_place(
            place, check_pir=True, check_symbol_infer=False
        )

    def test_check_grad(self):
        place = get_device_place()
        self.check_grad_with_place(
            place, ['X', 'Y'], 'Out', check_pir=True, check_prim_pir=True
        )


class TestElementwiseFminOpZeroSize(TestElementwiseFminOp):
    def init_shape(self):
        self.shape = [0, 9]


class TestElementwiseFminOpZeroSize1(TestElementwiseFminOp):
    def init_shape(self):
        self.shape = [9, 0]


@unittest.skipIf(
    not (core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA",
)
class TestElementwiseFminOp_Stride(OpTest):
    no_need_check_grad = True

    def setUp(self):
        self.op_type = "elementwise_fmin"
        self.python_api = paddle.fmin
        self.public_python_api = paddle.fmin
        self.transpose_api = paddle.transpose
        self.as_stride_api = paddle.as_strided
        self.init_dtype()
        self.init_input_output()

        self.inputs_stride = {
            'X': OpTest.np_dtype_to_base_dtype(self.x),
            'Y': OpTest.np_dtype_to_base_dtype(self.y_trans),
        }

        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(self.x),
            'Y': OpTest.np_dtype_to_base_dtype(self.y),
        }

        self.outputs = {'Out': self.out}

    def init_dtype(self):
        self.dtype = np.float64
        self.val_dtype = np.float64

    def test_check_output(self):
        place = get_device_place()
        self.check_strided_forward = True
        self.check_output(
            place,
        )

    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = np.fmin(self.x, self.y)
        self.perm = [1, 0]
        self.y_trans = np.transpose(self.y, self.perm)

    def test_check_gradient(self):
        pass


class TestElementwiseFminOp_Stride1(TestElementwiseFminOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.uniform(0.1, 1, [20, 2, 13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [20, 2, 13, 17]).astype(self.dtype)
        self.out = np.fmin(self.x, self.y)
        self.perm = [0, 1, 3, 2]
        self.y_trans = np.transpose(self.y, self.perm)


class TestElementwiseFminOp_Stride2(TestElementwiseFminOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.uniform(0.1, 1, [20, 2, 13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [20, 2, 13, 17]).astype(self.dtype)
        self.out = np.fmin(self.x, self.y)
        self.perm = [0, 2, 1, 3]
        self.y_trans = np.transpose(self.y, self.perm)


class TestElementwiseFminOp_Stride3(TestElementwiseFminOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.uniform(0.1, 1, [20, 2, 13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [20, 2, 13, 1]).astype(self.dtype)
        self.out = np.fmin(self.x, self.y)
        self.perm = [0, 1, 3, 2]
        self.y_trans = np.transpose(self.y, self.perm)


class TestElementwiseFminOp_Stride4(TestElementwiseFminOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.uniform(0.1, 1, [1, 2, 13, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [20, 2, 13, 1]).astype(self.dtype)
        self.out = np.fmin(self.x, self.y)
        self.perm = [1, 0, 2, 3]
        self.y_trans = np.transpose(self.y, self.perm)


class TestElementwiseFminOp_Stride5(TestElementwiseFminOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "as_stride"
        self.x = np.random.uniform(0.1, 1, [23, 10, 1, 17]).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [23, 2, 13, 20]).astype(self.dtype)
        self.y_trans = self.y
        self.y = self.y[:, 0:1, :, 0:1]
        self.out = np.fmin(self.x, self.y)
        self.shape_param = [23, 1, 13, 1]
        self.stride_param = [520, 260, 20, 1]


class TestElementwiseFminOp_Stride_ZeroDim1(TestElementwiseFminOp_Stride):
    def init_input_output(self):
        self.strided_input_type = "transpose"
        self.x = np.random.uniform(0.1, 1, []).astype(self.dtype)
        self.y = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.out = np.fmin(self.x, self.y)
        self.perm = [1, 0]
        self.y_trans = np.transpose(self.y, self.perm)


class TestElementwiseFminOp_Stride_ZeroSize1(TestElementwiseFminOp_Stride):
    def init_data(self):
        self.strided_input_type = "transpose"
        self.x = np.random.rand(1, 0, 2).astype('float32')
        self.y = np.random.rand(3, 0, 1).astype('float32')
        self.out = np.fmin(self.x, self.y)
        self.perm = [2, 1, 0]
        self.y_trans = np.transpose(self.y, self.perm)


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
