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

import os
import unittest

import gradient_checker
import numpy as np
from decorator_helper import prog_scope
from op_test import OpTest, convert_float_to_uint16, skip_check_grad_ci

import paddle
import paddle.distributed as dist
from paddle import base
from paddle.base import core
from paddle.pir_utils import IrGuard


class TestConcatOp(OpTest):
    def setUp(self):
        self.op_type = "concat"
        self.python_api = paddle.concat
        self.public_python_api = paddle.concat
        self.prim_op_type = "prim"
        self.dtype = self.get_dtype()
        self.init_test_data()
        self.if_enable_cinn()
        self.inputs = {'X': [('x0', self.x0), ('x1', self.x1), ('x2', self.x2)]}
        self.attrs = {'axis': self.axis}
        if self.axis < 0:
            self.actual_axis = self.axis + len(self.x0.shape)
            self.actual_axis = max(0, self.actual_axis)
        else:
            self.actual_axis = self.axis

        self.outputs = {
            'Out': np.concatenate(
                (self.x0, self.x1, self.x2), axis=self.actual_axis
            )
        }

    def get_dtype(self):
        return "float64"

    def test_check_output(self):
        if self.dtype == np.uint16:
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, check_pir=True)
        else:
            self.check_output(check_pir=True)

    def test_check_grad(self):
        if self.dtype == np.uint16:
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                ['x0'],
                'Out',
                check_prim=True,
                check_pir=True,
                check_prim_pir=True,
            )
            self.check_grad_with_place(
                place,
                ['x1'],
                'Out',
                check_prim=True,
                check_pir=True,
                check_prim_pir=True,
            )
            self.check_grad_with_place(
                place,
                ['x2'],
                'Out',
                check_prim=True,
                check_pir=True,
                check_prim_pir=True,
            )
        else:
            self.check_grad(
                ['x0'],
                'Out',
                check_prim=True,
                check_pir=True,
                check_prim_pir=True,
            )
            self.check_grad(
                ['x1'],
                'Out',
                check_prim=True,
                check_pir=True,
                check_prim_pir=True,
            )
            self.check_grad(
                ['x2'],
                'Out',
                check_prim=True,
                check_pir=True,
                check_prim_pir=True,
            )

    def init_test_data(self):
        if self.dtype == np.uint16:
            x0 = np.random.random((5, 1, 4, 5)).astype(np.float32)
            self.x0 = convert_float_to_uint16(x0)
            x1 = np.random.random((5, 2, 4, 5)).astype(np.float32)
            self.x1 = convert_float_to_uint16(x1)
            x2 = np.random.random((5, 3, 4, 5)).astype(np.float32)
            self.x2 = convert_float_to_uint16(x2)
        else:
            self.x0 = np.random.random((5, 1, 4, 5)).astype(self.dtype)
            self.x1 = np.random.random((5, 2, 4, 5)).astype(self.dtype)
            self.x2 = np.random.random((5, 3, 4, 5)).astype(self.dtype)
        self.axis = 1

    def if_enable_cinn(self):
        pass


class TestConcatOp2(TestConcatOp):
    def init_test_data(self):
        self.x0 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.x1 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.x2 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.axis = 1


@skip_check_grad_ci(
    reason="The function 'check_grad' for large inputs is too slow."
)
class TestConcatOp3(TestConcatOp):
    def init_test_data(self):
        self.x0 = np.random.random((1, 256, 170, 256)).astype(self.dtype)
        self.x1 = np.random.random((1, 128, 170, 256)).astype(self.dtype)
        self.x2 = np.random.random((1, 128, 170, 256)).astype(self.dtype)
        self.axis = 1

    def test_check_grad(self):
        pass


@skip_check_grad_ci(
    reason="This test will meet fetch error when there is a null grad. The detailed information is in PR#17015."
)
class TestConcatOp4(TestConcatOp):
    def init_test_data(self):
        self.x0 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.x1 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.x2 = np.random.random((0, 3, 4, 5)).astype(self.dtype)
        self.axis = 0

    def test_check_grad(self):
        pass


class TestConcatOp5(TestConcatOp):
    def init_test_data(self):
        self.x0 = np.random.random((5, 1, 4, 5)).astype(self.dtype)
        self.x1 = np.random.random((5, 2, 4, 5)).astype(self.dtype)
        self.x2 = np.random.random((5, 3, 4, 5)).astype(self.dtype)
        self.axis = -3


class TestConcatOp6(TestConcatOp):
    def setUp(self):
        self.op_type = "concat"
        self.dtype = self.get_dtype()
        self.python_api = paddle.concat
        self.public_python_api = paddle.concat
        self.init_test_data()
        self.if_enable_cinn()
        self.lod = [[20, 80]]
        self.out_lod = [[20, 80, 20, 80, 20, 80]]
        self.inputs = {
            'X': [
                ('x0', (self.x0, self.lod)),
                ('x1', (self.x1, self.lod)),
                ('x2', (self.x2, self.lod)),
            ]
        }
        self.attrs = {'axis': self.axis}
        if self.axis < 0:
            self.actual_axis = self.axis + len(self.x0.shape)
            self.actual_axis = max(0, self.actual_axis)
        else:
            self.actual_axis = self.axis
        out = np.concatenate((self.x0, self.x1, self.x2), axis=self.actual_axis)
        self.outputs = {'Out': (out, self.out_lod)}

    def if_enable_cinn(self):
        pass

    def test_check_output(self):
        self.check_output(check_pir=False)

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out', check_pir=False)
        self.check_grad(['x1'], 'Out', check_pir=False)
        self.check_grad(['x2'], 'Out', check_pir=False)

    def init_test_data(self):
        self.x0 = np.random.random([100]).astype(self.dtype)
        self.x1 = np.random.random([100]).astype(self.dtype)
        self.x2 = np.random.random([100]).astype(self.dtype)
        self.axis = 0


class TestConcatOp7(TestConcatOp):
    def setUp(self):
        self.op_type = "concat"
        self.python_api = paddle.concat
        self.public_python_api = paddle.concat
        self.prim_op_type = "prim"
        self.if_enable_cinn()
        self.dtype = self.get_dtype()
        self.init_test_data()
        self.inputs = {'X': [('x0', self.x0), ('x1', self.x1), ('x2', self.x2)]}
        self.attrs = {'axis': self.axis}
        if self.axis < 0:
            self.actual_axis = self.axis + len(self.x0.shape)
            self.actual_axis = max(0, self.actual_axis)
        else:
            self.actual_axis = self.axis

        self.outputs = {
            'Out': np.concatenate(
                (self.x0, self.x1, self.x2), axis=self.actual_axis
            )
        }

    def if_enable_cinn(self):
        pass

    def get_dtype(self):
        return "float64"

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['x0'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )
        self.check_grad(
            ['x1'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )
        self.check_grad(
            ['x2'],
            'Out',
            check_prim=True,
            check_pir=True,
            check_prim_pir=True,
        )

    def init_test_data(self):
        if self.dtype == np.uint16:
            x0 = np.random.random((5, 1, 4, 5)).astype(np.float32)
            self.x0 = convert_float_to_uint16(x0)
            x1 = np.random.random((5, 2, 4, 5)).astype(np.float32)
            self.x1 = convert_float_to_uint16(x1)
            x2 = np.random.random((5, 3, 4, 5)).astype(np.float32)
            self.x2 = convert_float_to_uint16(x2)
        else:
            self.x0 = np.random.random((5, 1, 4, 5)).astype(self.dtype)
            self.x1 = np.random.random((5, 2, 4, 5)).astype(self.dtype)
            self.x2 = np.random.random((5, 3, 4, 5)).astype(self.dtype)
        self.axis = 1


def create_test_AxisTensor(parent):
    class TestConcatAxisTensor(parent):
        def setUp(self):
            self.op_type = "concat"
            self.python_api = paddle.concat
            self.public_python_api = paddle.concat
            self.dtype = self.get_dtype()
            self.init_test_data()
            self.inputs = {
                'X': [('x0', self.x0), ('x1', self.x1), ('x2', self.x2)],
                'AxisTensor': np.array([self.axis]).astype("int32"),
            }
            self.attrs = {}

            if self.axis < 0:
                self.actual_axis = self.axis + len(self.x0.shape)
                self.actual_axis = max(0, self.actual_axis)
            else:
                self.actual_axis = self.axis

            self.outputs = {
                'Out': np.concatenate(
                    (self.x0, self.x1, self.x2), axis=self.actual_axis
                )
            }

        def test_check_output(self):
            if self.dtype == np.uint16:
                place = core.CUDAPlace(0)
                self.check_output_with_place(
                    place, check_pir=True, check_symbol_infer=False
                )
            else:
                self.check_output(check_pir=True, check_symbol_infer=False)

        def test_check_grad(self):
            if (
                parent.__name__ == 'TestConcatOp4'
                or parent.__name__ == 'TestConcatOp3'
            ):
                return
            if self.dtype == np.uint16:
                place = core.CUDAPlace(0)
                self.check_grad_with_place(place, ['x0'], 'Out', check_pir=True)
                self.check_grad_with_place(place, ['x1'], 'Out', check_pir=True)
                self.check_grad_with_place(place, ['x2'], 'Out', check_pir=True)
            else:
                self.check_grad(['x0'], 'Out', check_pir=True)
                self.check_grad(['x1'], 'Out', check_pir=True)
                self.check_grad(['x2'], 'Out', check_pir=True)

    cls_name = "{}_{}".format(parent.__name__, "AxisTensor")
    TestConcatAxisTensor.__name__ = cls_name
    globals()[cls_name] = TestConcatAxisTensor


create_test_AxisTensor(TestConcatOp)
create_test_AxisTensor(TestConcatOp2)
create_test_AxisTensor(TestConcatOp3)
create_test_AxisTensor(TestConcatOp4)
create_test_AxisTensor(TestConcatOp5)
create_test_AxisTensor(TestConcatOp6)

# ----------------Concat Fp16----------------


def create_test_fp16(parent):
    class TestConcatFp16(parent):
        def setUp(self):
            self.op_type = "concat"
            self.prim_op_type = "prim"
            self.python_api = paddle.concat
            self.public_python_api = paddle.concat
            self.enable_cinn = False
            self.dtype = self.get_dtype()
            self.init_test_data()
            self.inputs = {
                'X': [('x0', self.x0), ('x1', self.x1), ('x2', self.x2)]
            }
            self.attrs = {'axis': self.axis}
            if self.axis < 0:
                self.actual_axis = self.axis + len(self.x0.shape)
                self.actual_axis = max(0, self.actual_axis)
            else:
                self.actual_axis = self.axis

            self.outputs = {
                'Out': np.concatenate(
                    (self.x0, self.x1, self.x2), axis=self.actual_axis
                )
            }

        def test_check_grad(self):
            if (
                parent.__name__ == 'TestConcatOp4'
                or parent.__name__ == 'TestConcatOp3'
            ):
                return
            if self.dtype == np.uint16:
                place = core.CUDAPlace(0)
                self.check_grad_with_place(
                    place,
                    ['x0'],
                    'Out',
                    check_pir=True,
                    check_prim=True,
                    check_prim_pir=True,
                )
                self.check_grad_with_place(
                    place,
                    ['x1'],
                    'Out',
                    check_pir=True,
                    check_prim=True,
                    check_prim_pir=True,
                )
                self.check_grad_with_place(
                    place,
                    ['x2'],
                    'Out',
                    check_pir=True,
                    check_prim=True,
                    check_prim_pir=True,
                )
            else:
                self.check_grad(
                    ['x0'],
                    'Out',
                    check_pir=True,
                    check_prim=True,
                    check_prim_pir=True,
                )
                self.check_grad(
                    ['x1'],
                    'Out',
                    check_pir=True,
                    check_prim=True,
                    check_prim_pir=True,
                )
                self.check_grad(
                    ['x2'],
                    'Out',
                    check_pir=True,
                    check_prim=True,
                    check_prim_pir=True,
                )

        def get_dtype(self):
            return np.float16

    cls_name = "{}_{}".format(parent.__name__, "Fp16")
    TestConcatFp16.__name__ = cls_name
    globals()[cls_name] = TestConcatFp16


create_test_fp16(TestConcatOp)
create_test_fp16(TestConcatOp2)
create_test_fp16(TestConcatOp3)
create_test_fp16(TestConcatOp4)

create_test_fp16(TestConcatOp5)

create_test_fp16(TestConcatOp6)


# ----------------Concat Bf16----------------
def create_test_bf16(parent):
    @unittest.skipIf(
        not paddle.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestConcatBf16(parent):
        def setUp(self):
            self.op_type = "concat"
            self.prim_op_type = "prim"
            self.python_api = paddle.concat
            self.public_python_api = paddle.concat
            self.enable_cinn = False
            self.dtype = self.get_dtype()
            self.init_test_data()
            self.inputs = {
                'X': [('x0', self.x0), ('x1', self.x1), ('x2', self.x2)]
            }
            self.attrs = {'axis': self.axis}
            if self.axis < 0:
                self.actual_axis = self.axis + len(self.x0.shape)
                self.actual_axis = max(0, self.actual_axis)
            else:
                self.actual_axis = self.axis

            self.outputs = {
                'Out': np.concatenate(
                    (self.x0, self.x1, self.x2), axis=self.actual_axis
                )
            }

        def test_check_grad(self):
            if (
                parent.__name__ == 'TestConcatOp4'
                or parent.__name__ == 'TestConcatOp3'
            ):
                return
            if self.dtype == np.uint16:
                place = core.CUDAPlace(0)
                self.check_grad_with_place(
                    place,
                    ['x0'],
                    'Out',
                    check_pir=True,
                    check_prim=True,
                    check_prim_pir=True,
                )
                self.check_grad_with_place(
                    place,
                    ['x1'],
                    'Out',
                    check_pir=True,
                    check_prim=True,
                    check_prim_pir=True,
                )
                self.check_grad_with_place(
                    place,
                    ['x2'],
                    'Out',
                    check_pir=True,
                    check_prim=True,
                    check_prim_pir=True,
                )
            else:
                self.check_grad(
                    ['x0'],
                    'Out',
                    check_pir=True,
                    check_prim=True,
                    check_prim_pir=True,
                )
                self.check_grad(
                    ['x1'],
                    'Out',
                    check_pir=True,
                    check_prim=True,
                    check_prim_pir=True,
                )
                self.check_grad(
                    ['x2'],
                    'Out',
                    check_pir=True,
                    check_prim=True,
                    check_prim_pir=True,
                )

        def get_dtype(self):
            return np.uint16

        def if_enable_cinn(self):
            self.enable_cinn = False

    cls_name = "{}_{}".format(parent.__name__, "Bf16")
    TestConcatBf16.__name__ = cls_name
    globals()[cls_name] = TestConcatBf16


# add all unit test maybe timeout.
create_test_bf16(TestConcatOp)
create_test_bf16(TestConcatOp2)
# create_test_bf16(TestConcatOp3)
create_test_bf16(TestConcatOp4)
# create_test_bf16(TestConcatOp5)
# create_test_bf16(TestConcatOp6)


class TestConcatOpError(unittest.TestCase):
    def test_errors(self):
        paddle.enable_static()
        with paddle.base.program_guard(
            paddle.base.Program(), paddle.base.Program()
        ):
            # The input type of concat_op should be list.

            x1 = paddle.static.data(shape=[-1, 4], dtype='int32', name='x1')
            paddle.concat(x1)

            # The item in input must be Variable.
            x2 = base.create_lod_tensor(
                np.array([[-1]]), [[1]], base.CPUPlace()
            )
            x3 = base.create_lod_tensor(
                np.array([[-1]]), [[1]], base.CPUPlace()
            )
            self.assertRaises(TypeError, paddle.concat, [x2])
            # The input dtype of concat_op must be float16, float32, float64, int32, int64.

            x4 = paddle.static.data(shape=[-1, 4], dtype='uint8', name='x4')
            x5 = paddle.static.data(shape=[-1, 4], dtype='uint8', name='x5')
            self.assertRaises(TypeError, paddle.concat, [x4, x5])
            x6 = paddle.static.data(shape=[-1, 4], dtype='float16', name='x6')
            x7 = paddle.static.data(shape=[-1, 4], dtype='float16', name='x7')
            x8 = paddle.static.data(shape=[-1, 4], dtype='float32', name='x8')
            paddle.concat([x6, x7])

            # The type of axis in concat_op should be int or Variable.
            def test_axis_type():
                paddle.concat([x6, x7], 3.2)

            self.assertRaises(TypeError, test_axis_type)

            def test_input_same_dtype():
                paddle.concat([x7, x8])

            self.assertRaises(TypeError, test_input_same_dtype)
        paddle.disable_static()


class TestConcatAPI(unittest.TestCase):

    def test_base_api(self):
        paddle.enable_static()
        with paddle.base.program_guard(paddle.base.Program()):
            x_1 = paddle.static.data(
                shape=[None, 1, 4, 5], dtype='int32', name='x_1'
            )
            paddle.concat([x_1, x_1], 0)

            input_2 = np.random.random([2, 1, 4, 5]).astype("int32")
            input_3 = np.random.random([2, 2, 4, 5]).astype("int32")
            x_2 = paddle.static.data(
                shape=[2, 1, 4, 5], dtype='int32', name='x_2'
            )
            x_3 = paddle.static.data(
                shape=[2, 2, 4, 5], dtype='int32', name='x_3'
            )
            positive_1_int32 = paddle.tensor.fill_constant([1], "int32", 1)
            positive_1_int64 = paddle.tensor.fill_constant([1], "int64", 1)
            out_1 = paddle.concat([x_2, x_3], axis=1)
            out_2 = paddle.concat([x_2, x_3], axis=positive_1_int32)
            out_3 = paddle.concat([x_2, x_3], axis=positive_1_int64)

            exe = base.Executor(place=base.CPUPlace())
            [res_1, res_2, res_3] = exe.run(
                paddle.static.default_main_program(),
                feed={"x_1": input_2, "x_2": input_2, "x_3": input_3},
                fetch_list=[out_1, out_2, out_3],
            )
            np.testing.assert_array_equal(
                res_1, np.concatenate((input_2, input_3), axis=1)
            )
            np.testing.assert_array_equal(
                res_2, np.concatenate((input_2, input_3), axis=1)
            )
            np.testing.assert_array_equal(
                res_3, np.concatenate((input_2, input_3), axis=1)
            )

    def test_api(self):
        paddle.enable_static()
        with paddle.base.program_guard(paddle.base.Program()):
            x_1 = paddle.static.data(
                shape=[None, 1, 4, 5], dtype='int32', name='x_1'
            )
            paddle.concat([x_1, x_1], 0)

            input_2 = np.random.random([2, 1, 4, 5]).astype("int32")
            input_3 = np.random.random([2, 2, 4, 5]).astype("int32")
            x_2 = paddle.static.data(
                shape=[2, 1, 4, 5], dtype='int32', name='x_2'
            )
            x_3 = paddle.static.data(
                shape=[2, 2, 4, 5], dtype='int32', name='x_3'
            )
            positive_1_int32 = paddle.tensor.fill_constant([1], "int32", 1)
            positive_1_int64 = paddle.tensor.fill_constant([1], "int64", 1)
            negative_int64 = paddle.tensor.fill_constant([1], "int64", -3)
            out_1 = paddle.concat(x=[x_2, x_3], axis=1)
            out_2 = paddle.concat(x=[x_2, x_3], axis=positive_1_int32)
            out_3 = paddle.concat(x=[x_2, x_3], axis=positive_1_int64)
            out_4 = paddle.concat(x=[x_2, x_3], axis=negative_int64)

            exe = paddle.static.Executor(place=paddle.CPUPlace())
            [res_1, res_2, res_3, res_4] = exe.run(
                paddle.static.default_main_program(),
                feed={"x_1": input_2, "x_2": input_2, "x_3": input_3},
                fetch_list=[out_1, out_2, out_3, out_4],
            )
            np.testing.assert_array_equal(
                res_1, np.concatenate((input_2, input_3), axis=1)
            )
            np.testing.assert_array_equal(
                res_2, np.concatenate((input_2, input_3), axis=1)
            )
            np.testing.assert_array_equal(
                res_3, np.concatenate((input_2, input_3), axis=1)
            )
            np.testing.assert_array_equal(
                res_4, np.concatenate((input_2, input_3), axis=1)
            )

    def test_imperative(self):
        in1 = np.array([[1, 2, 3], [4, 5, 6]])
        in2 = np.array([[11, 12, 13], [14, 15, 16]])
        in3 = np.array([[21, 22], [23, 24]])
        paddle.disable_static()
        x1 = paddle.to_tensor(in1)
        x2 = paddle.to_tensor(in2)
        x3 = paddle.to_tensor(in3)
        out1 = paddle.concat([x1, x2, x3], axis=-1)
        out2 = paddle.concat(x=[x1, x2], axis=0)
        np_out1 = np.concatenate([in1, in2, in3], axis=-1)
        np_out2 = np.concatenate([in1, in2], axis=0)
        paddle.enable_static()
        self.assertEqual((out1.numpy() == np_out1).all(), True)
        self.assertEqual((out2.numpy() == np_out2).all(), True)

    def test_errors(self):
        with paddle.base.program_guard(
            paddle.base.Program(), paddle.base.Program()
        ):
            # The item in input must be Variable.
            x2 = base.create_lod_tensor(
                np.array([[-1]]), [[1]], base.CPUPlace()
            )
            x3 = base.create_lod_tensor(
                np.array([[-1]]), [[1]], base.CPUPlace()
            )
            self.assertRaises(TypeError, paddle.concat, [x2])
            # The input dtype of concat_op must be float16, float32, float64, int32, int64.
            x4 = paddle.static.data(shape=[4], dtype='uint8', name='x4')
            x5 = paddle.static.data(shape=[4], dtype='uint8', name='x5')
            self.assertRaises(TypeError, paddle.concat, [x4, x5])

            # The type of axis in concat_op should be int or Variable.
            x6 = paddle.static.data(shape=[-1, 4], dtype='float16', name='x6')
            x7 = paddle.static.data(shape=[-1, 4], dtype='float16', name='x7')
            x8 = paddle.static.data(shape=[-1, 4], dtype='float32', name='x8')

            def test_axis_type():
                paddle.concat([x6, x7], 3.2)

            self.assertRaises(TypeError, test_axis_type)

            def test_input_same_dtype():
                paddle.concat([x7, x8])

            self.assertRaises(TypeError, test_input_same_dtype)


class TestConcatAPIWithDenseTensorArray(unittest.TestCase):
    """
    Test concat api when the input(x) is a DenseTensorArray.
    """

    def setUp(self):
        self.axis = 1
        self.python = paddle.concat
        self.iter_num = 3
        self.input_shape = [2, 3]
        self.x = np.random.random(self.input_shape).astype("float32")
        self.place = (
            base.CUDAPlace(0)
            if base.is_compiled_with_cuda()
            else base.CPUPlace()
        )

    def set_program(self, use_base_api):
        paddle.enable_static()
        if use_base_api:
            self.program = paddle.base.Program()
            with paddle.base.program_guard(self.program):
                input = paddle.assign(self.x)
                tensor_array = paddle.tensor.create_array(dtype='float32')
                zero = paddle.tensor.fill_constant(
                    shape=[1], value=0, dtype="int64"
                )

                for i in range(self.iter_num):
                    paddle.tensor.array_write(input, zero + i, tensor_array)

                self.out_var = paddle.concat(tensor_array, axis=self.axis)
        else:
            self.program = paddle.base.Program()
            with paddle.base.program_guard(self.program):
                input = paddle.assign(self.x)
                tensor_array = paddle.tensor.create_array(
                    dtype='float32'
                )  # Api create_array is not supported in paddle 2.0 yet.
                zero = paddle.zeros(shape=[1], dtype="int64")

                for i in range(self.iter_num):
                    # Api array_write is not supported in paddle 2.0 yet.
                    paddle.tensor.array_write(input, zero + i, tensor_array)

                self.out_var = paddle.concat(tensor_array, axis=self.axis)

    def test_base_api(self):
        self._run_static_mode(use_base_api=True)

    def test_paddle_api(self):
        self._run_static_mode(use_base_api=False)

    def _run_static_mode(self, use_base_api):
        self.set_program(use_base_api)
        self.assertTrue(self.out_var.shape[self.axis] == -1)
        exe = base.Executor(self.place)
        res = exe.run(self.program, fetch_list=self.out_var)
        np.testing.assert_array_equal(
            res[0], np.concatenate([self.x] * self.iter_num, axis=self.axis)
        )


class TestConcatDoubleGradCheck(unittest.TestCase):
    def concat_wrapper(self, x):
        return paddle.concat(x)

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not include -1.
        eps = 0.005
        dtype = np.float32

        data1 = paddle.static.data('data1', [2, 3], dtype)
        data1.persistable = True
        data1.stop_gradient = False
        data2 = paddle.static.data('data2', [2, 3], dtype)
        data2.persistable = True
        data2.stop_gradient = False
        out = paddle.concat([data1, data2])
        data1_arr = np.random.uniform(-1, 1, data1.shape).astype(dtype)
        data2_arr = np.random.uniform(-1, 1, data2.shape).astype(dtype)
        gradient_checker.double_grad_check(
            [data1, data2],
            out,
            x_init=[data1_arr, data2_arr],
            place=place,
            eps=eps,
        )
        gradient_checker.double_grad_check_for_dygraph(
            self.concat_wrapper,
            [data1, data2],
            out,
            x_init=[data1_arr, data2_arr],
            place=place,
        )

    def test_grad(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestConcatTripleGradCheck(unittest.TestCase):
    def concat_wrapper(self, x):
        return paddle.concat(x, 1)

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not include -1.
        eps = 0.005
        dtype = np.float32

        data1 = paddle.static.data('data1', [2, 3, 4], dtype)
        data1.persistable = True
        data1.stop_gradient = False
        data2 = paddle.static.data('data2', [2, 3, 4], dtype)
        data2.persistable = True
        data2.stop_gradient = False
        out = paddle.concat([data1, data2], 1)
        data1_arr = np.random.uniform(-1, 1, data1.shape).astype(dtype)
        data2_arr = np.random.uniform(-1, 1, data2.shape).astype(dtype)
        gradient_checker.triple_grad_check(
            [data1, data2],
            out,
            x_init=[data1_arr, data2_arr],
            place=place,
            eps=eps,
        )
        gradient_checker.triple_grad_check_for_dygraph(
            self.concat_wrapper,
            [data1, data2],
            out,
            x_init=[data1_arr, data2_arr],
            place=place,
        )

    def test_grad(self):
        paddle.enable_static()
        places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            places.append(base.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestConcatOpAutoParallel(OpTest):
    def setUp(self):
        self.op_type = "concat"
        self.python_api = paddle.concat
        self.public_python_api = paddle.concat
        self.prim_op_type = "prim"
        self.dtype = self.get_dtype()
        self.init_test_data()
        self.if_enable_cinn()
        self.init_inputs()
        self.attrs = {'axis': self.axis}
        if self.axis < 0:
            self.actual_axis = self.axis + len(self.x0.shape)
            self.actual_axis = max(0, self.actual_axis)
        else:
            self.actual_axis = self.axis

        self.outputs = {
            'Out': np.concatenate(
                (self.x0, self.x1, self.x2), axis=self.actual_axis
            )
        }

    def get_dtype(self):
        return "float64"

    def init_inputs(self):
        self.inputs = {'X': [('x0', self.x0), ('x1', self.x1), ('x2', self.x2)]}
        self.placements = {
            'X': [
                ('x0', [dist.Shard(2)]),
                ('x1', [dist.Shard(2)]),
                ('x2', [dist.Shard(2)]),
            ]
        }

    def test_check_grad(self):
        self.check_grad(
            ['x0'],
            'Out',
            check_auto_parallel=True,
        )
        self.check_grad(
            ['x0', 'x1', 'x2'],
            'Out',
            check_auto_parallel=True,
        )

    def init_test_data(self):
        if self.dtype == np.uint16:
            x0 = np.random.random((16, 4, 4)).astype(np.float32)
            self.x0 = convert_float_to_uint16(x0)
            x1 = np.random.random((64, 4, 4)).astype(np.float32)
            self.x1 = convert_float_to_uint16(x1)
            x2 = np.random.random((16, 4, 4)).astype(np.float32)
            self.x2 = convert_float_to_uint16(x2)
        else:
            self.x0 = np.random.random((16, 4, 4)).astype(self.dtype)
            self.x1 = np.random.random((64, 4, 4)).astype(self.dtype)
            self.x2 = np.random.random((16, 4, 4)).astype(self.dtype)
        self.axis = 0

    def if_enable_cinn(self):
        pass


class TestConcatOpErrorWithPir(unittest.TestCase):

    def test_errors_with_pir(self):
        paddle.enable_static()
        with paddle.base.program_guard(
            paddle.base.Program(), paddle.base.Program()
        ):
            # The type of axis in concat_op should be int or Variable.
            x6 = paddle.static.data(shape=[-1, 4], dtype='float32', name='x6')
            x7 = paddle.static.data(shape=[-1, 4], dtype='float32', name='x7')
            x8 = paddle.static.data(shape=[-1, 4], dtype='float64', name='x8')

            def test_axis_type():
                paddle.concat([x6, x7], 3.2)

            self.assertRaises(TypeError, test_axis_type)

            # The input dtype must be same.
            def test_input_same_dtype():
                paddle.concat([x7, x8])

            self.assertRaises(TypeError, test_input_same_dtype)

    def test_empty_inputs_dygraph(self):
        paddle.disable_static()
        with self.assertRaisesRegex(ValueError, "but got empty list"):
            paddle.concat([])

    def test_empty_inputs_static(self):
        with IrGuard(), paddle.base.program_guard(
            paddle.base.Program(), paddle.base.Program()
        ):
            with self.assertRaisesRegex(ValueError, "but got empty list"):
                paddle.concat([], axis=0)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
