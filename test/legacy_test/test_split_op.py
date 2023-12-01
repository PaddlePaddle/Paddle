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
from paddle import base
from paddle.base import Program, core, program_guard
from paddle.pir_utils import test_with_pir_api


class TestSplitOp(OpTest):
    def setUp(self):
        self.python_api = paddle.split
        self.public_python_api = paddle.split
        self.python_out_sig = ['out0', 'out1', 'out2']
        self._set_op_type()
        self.prim_op_type = "prim"
        self.dtype = self.get_dtype()
        axis = 1
        if self.dtype == np.uint16:
            x = np.random.random((4, 5, 6)).astype(np.float32)
            out = np.split(x, [2, 3], axis)
            self.inputs = {'X': convert_float_to_uint16(x)}
            self.outputs = {
                'Out': [
                    ('out%d' % i, convert_float_to_uint16(out[i]))
                    for i in range(len(out))
                ]
            }
        else:
            x = np.random.random((4, 5, 6)).astype(self.dtype)
            out = np.split(x, [2, 3], axis)
            self.inputs = {'X': x}
            self.outputs = {
                'Out': [('out%d' % i, out[i]) for i in range(len(out))]
            }
        self.attrs = {'axis': axis, 'sections': [2, 1, 2]}

    def get_dtype(self):
        return "float64"

    def _set_op_type(self):
        self.op_type = "split"

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            ['out0', 'out1', 'out2'],
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
        )


# test with attr(num)
class TestSplitWithNumOp(OpTest):
    def setUp(self):
        self.python_api = paddle.split
        self.public_python_api = paddle.split
        self.python_out_sig = ['out0', 'out1', 'out2']
        self._set_op_type()
        self.prim_op_type = "prim"
        self.dtype = self.get_dtype()
        self.init_data()
        self.attrs = {
            'axis': self.axis,
            'sections': self.sections,
            'num': self.num,
        }
        if self.dtype == np.uint16:
            self.inputs = {'X': convert_float_to_uint16(self.x)}
            out = np.split(self.x, self.indices_or_sections, self.axis)
            self.outputs = {
                'Out': [
                    ('out%d' % i, convert_float_to_uint16(out[i]))
                    for i in range(len(out))
                ]
            }
        else:
            self.inputs = {'X': self.x}
            out = np.split(self.x, self.indices_or_sections, self.axis)
            self.outputs = {
                'Out': [('out%d' % i, out[i]) for i in range(len(out))]
            }

    def init_data(self):
        if self.dtype == np.uint16:
            self.x = np.random.random((4, 5, 6)).astype(np.float32)
        else:
            self.x = np.random.random((4, 5, 6)).astype(self.dtype)
        self.axis = 2
        self.sections = []
        self.num = 3
        self.indices_or_sections = 3

    def get_dtype(self):
        return "float64"

    def _set_op_type(self):
        self.op_type = "split"

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            ['out0', 'out1', 'out2'],
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
        )


# attr(axis) is Tensor
class TestSplitOp_AxisTensor(OpTest):
    def setUp(self):
        self.python_api = paddle.split
        self.python_out_sig = ['out0', 'out1', 'out2']
        self._set_op_type()
        self.dtype = self.get_dtype()
        self.init_data()
        self.inputs = {
            'X': self.x,
            'AxisTensor': np.array([self.axis]).astype("int32"),
        }
        self.attrs = {'sections': self.sections, 'num': self.num}

        out = np.split(self.x, self.indices_or_sections, self.axis)
        self.outputs = {'Out': [('out%d' % i, out[i]) for i in range(len(out))]}

    def init_data(self):
        self.x = np.random.random((4, 5, 6)).astype(self.dtype)
        self.axis = 2
        self.sections = []
        self.num = 3
        self.indices_or_sections = 3

    def get_dtype(self):
        return "float64"

    def _set_op_type(self):
        self.op_type = "split"

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], ['out0', 'out1', 'out2'], check_pir=True)


# attr(sections) is list containing Tensor
class TestSplitOp_SectionsTensor(OpTest):
    def setUp(self):
        self.python_api = paddle.split
        self.python_out_sig = ['out0', 'out1', 'out2']
        self._set_op_type()
        self.dtype = self.get_dtype()
        self.init_data()
        self.inputs = {'X': self.x}

        sections_tensor = []
        for index, ele in enumerate(self.sections):
            sections_tensor.append(
                ("x" + str(index), np.ones(1).astype('int32') * ele)
            )

        self.inputs['SectionsTensorList'] = sections_tensor

        self.attrs = {
            'axis': self.axis,
            'sections': self.sections_infer,
            'num': self.num,
        }

        out = np.split(self.x, self.indices_or_sections, self.axis)
        self.outputs = {'Out': [('out%d' % i, out[i]) for i in range(len(out))]}

    def init_data(self):
        self.x = np.random.random((4, 5, 6)).astype(self.dtype)
        self.axis = 1
        self.sections = [2, 1, 2]
        self.sections_infer = [-1, -1, -1]
        self.num = 0
        self.indices_or_sections = [2, 3]

    def get_dtype(self):
        return "float64"

    def _set_op_type(self):
        self.op_type = "split"

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(['X'], ['out0', 'out1', 'out2'], check_pir=True)


class TestSplitOp_unk_section(OpTest):
    def setUp(self):
        self.python_api = paddle.split
        self.public_python_api = paddle.split
        self.python_out_sig = ['out0', 'out1', 'out2']
        self._set_op_type()
        self.prim_op_type = "prim"
        self.dtype = self.get_dtype()
        self.init_data()
        self.inputs = {'X': self.x}
        self.attrs = {
            'axis': self.axis,
            'sections': self.sections,
            'num': self.num,
        }

        out = np.split(self.x, self.indices_or_sections, self.axis)
        self.outputs = {'Out': [('out%d' % i, out[i]) for i in range(len(out))]}

    def init_data(self):
        self.x = np.random.random((4, 5, 6)).astype(self.dtype)
        self.axis = 2
        self.sections = [2, 1, -1]
        self.num = 0
        self.indices_or_sections = [2, 3]

    def get_dtype(self):
        return "float64"

    def _set_op_type(self):
        self.op_type = "split"

    def test_check_output(self):
        self.check_output(check_pir=True)

    def test_check_grad(self):
        self.check_grad(
            ['X'],
            ['out0', 'out1', 'out2'],
            check_prim=True,
            check_prim_pir=True,
            check_pir=True,
        )


class TestSplitByrefOp(OpTest):
    def _set_op_type(self):
        self.op_type = "split_byref"


# ----------------Split Fp16----------------


def create_test_fp16(parent):
    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestSplitFP16Op(parent):
        def get_dtype(self):
            return np.float16

    cls_name = "{}_{}".format(parent.__name__, "FP16Op")
    TestSplitFP16Op.__name__ = cls_name
    globals()[cls_name] = TestSplitFP16Op


create_test_fp16(TestSplitOp)
create_test_fp16(TestSplitWithNumOp)

# ----------------Split Bf16----------------


def create_test_bf16(parent):
    @unittest.skipIf(
        not core.is_compiled_with_cuda()
        or not core.is_bfloat16_supported(core.CUDAPlace(0)),
        "core is not compiled with CUDA or not support bfloat16",
    )
    class TestSplitBF16Op(parent):
        def get_dtype(self):
            return np.uint16

        def test_check_output(self):
            place = core.CUDAPlace(0)
            self.check_output_with_place(place)

        def test_check_grad(self):
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place,
                ['X'],
                'out2',
                check_prim=True,
                check_prim_pir=True,
                check_pir=True,
            )

    cls_name = "{}_{}".format(parent.__name__, "BF16Op")
    TestSplitBF16Op.__name__ = cls_name
    globals()[cls_name] = TestSplitBF16Op


create_test_bf16(TestSplitOp)
create_test_bf16(TestSplitWithNumOp)


class TestSplitAPI(unittest.TestCase):
    @test_with_pir_api
    def test_api(self):
        with paddle.static.program_guard(paddle.static.Program()):
            input_1 = np.random.random([4, 5, 6]).astype("int32")
            positive_1_int32 = paddle.tensor.fill_constant([1], "int32", 1)
            positive_1_int64 = paddle.tensor.fill_constant([1], "int64", 1)
            positive_2_int64 = paddle.tensor.fill_constant([1], "int64", 2)
            x_1 = paddle.static.data(shape=[4, 5, 6], dtype='int32', name='x_1')
            x_2 = paddle.static.data(
                shape=[4, 5, None], dtype='int32', name='x_2'
            )

            out_0, out_1, out_2 = paddle.split(
                x=x_1,
                num_or_sections=[positive_2_int64, positive_1_int32, -1],
                axis=positive_1_int64,
            )

            out_3, out_4, out_5 = paddle.split(
                x=x_1, num_or_sections=[2, 1, 2], axis=positive_1_int32
            )
            paddle.split(x=x_2, num_or_sections=2, axis=2)

            exe = base.Executor(place=base.CPUPlace())
            [res_0, res_1, res_2, res_3, res_4, res_5] = exe.run(
                paddle.static.default_main_program(),
                feed={"x_1": input_1, "x_2": input_1},
                fetch_list=[out_0, out_1, out_2, out_3, out_4, out_5],
            )

            out = np.split(input_1, [2, 3], 1)
            np.testing.assert_array_equal(res_0, out[0])
            np.testing.assert_array_equal(res_1, out[1])
            np.testing.assert_array_equal(res_2, out[2])
            np.testing.assert_array_equal(res_3, out[0])
            np.testing.assert_array_equal(res_4, out[1])
            np.testing.assert_array_equal(res_5, out[2])


class TestSplitOpError(unittest.TestCase):
    def test_errors(self):
        paddle.enable_static()
        with program_guard(Program(), Program()):
            # The type of axis in split_op should be int or Variable.
            def test_axis_type():
                x6 = paddle.static.data(
                    shape=[-1, 4], dtype='float16', name='x3'
                )
                paddle.split(x=x6, num_or_sections=2, axis=3.2)

            self.assertRaises(TypeError, test_axis_type)

            # The type of axis in split_op should be int or Variable.
            def test_axis_variable_type():
                x9 = paddle.static.data(
                    shape=[-1, 4], dtype='float16', name='x9'
                )
                x10 = paddle.static.data(
                    shape=[-1, 1], dtype='float16', name='x10'
                )
                paddle.split(x=x9, num_or_sections=2, axis=x10)

            self.assertRaises(TypeError, test_axis_variable_type)

            # The type of num_or_sections in split_op should be int, tuple or list.
            def test_num_or_sections_type():
                x6 = paddle.static.data(
                    shape=[-1, 4], dtype='float16', name='x4'
                )
                paddle.split(x=x6, num_or_sections=2.1, axis=3)

            self.assertRaises(TypeError, test_num_or_sections_type)

            def test_num_or_sections_type_tensor():
                x7 = paddle.static.data(
                    shape=[-1, 4], dtype='float16', name='x5'
                )
                paddle.split(input=x7, num_or_sections=2.1, dim=3)

            self.assertRaises(TypeError, test_num_or_sections_type_tensor)

            def test_axis_type_tensor():
                x8 = paddle.static.data(
                    shape=[-1, 4], dtype='float16', name='x6'
                )
                paddle.split(input=x8, num_or_sections=2, dim=3.2)

            self.assertRaises(TypeError, test_axis_type_tensor)
        paddle.disable_static()

        with paddle.base.dygraph.guard():

            def test_0_num_tensor():
                x = paddle.uniform([1, 1, 1], dtype='float32')
                paddle.split(x, num_or_sections=0)

            self.assertRaises(ValueError, test_0_num_tensor)


class API_TestSplit(unittest.TestCase):
    @test_with_pir_api
    def test_out(self):
        with base.program_guard(base.Program(), base.Program()):
            data1 = paddle.static.data(
                'data1', shape=[4, 6, 6], dtype='float64'
            )
            data2 = paddle.static.data('data2', shape=[1], dtype='int32')
            x0, x1, x2 = paddle.split(data1, num_or_sections=3, axis=data2)
            place = base.CPUPlace()
            exe = base.Executor(place)
            input1 = np.random.random([4, 6, 6]).astype('float64')
            input2 = np.array([2]).astype('int32')
            (
                r0,
                r1,
                r2,
            ) = exe.run(
                feed={"data1": input1, "data2": input2}, fetch_list=[x0, x1, x2]
            )
            ex_x0, ex_x1, ex_x2 = np.split(input1, 3, axis=2)
            np.testing.assert_allclose(ex_x0, r0, rtol=1e-05)
            np.testing.assert_allclose(ex_x1, r1, rtol=1e-05)
            np.testing.assert_allclose(ex_x2, r2, rtol=1e-05)


class API_TestSplit2(unittest.TestCase):
    @test_with_pir_api
    def test_out(self):
        with base.program_guard(base.Program(), base.Program()):
            data1 = paddle.static.data(
                'data1', shape=[4, 6, 6], dtype='float64'
            )
            x0, x1, x2 = paddle.split(data1, num_or_sections=3, axis=2)
            place = base.CPUPlace()
            exe = base.Executor(place)
            input1 = np.random.random([4, 6, 6]).astype('float64')
            (
                r0,
                r1,
                r2,
            ) = exe.run(feed={"data1": input1}, fetch_list=[x0, x1, x2])
            ex_x0, ex_x1, ex_x2 = np.split(input1, 3, axis=2)
            np.testing.assert_allclose(ex_x0, r0, rtol=1e-05)
            np.testing.assert_allclose(ex_x1, r1, rtol=1e-05)
            np.testing.assert_allclose(ex_x2, r2, rtol=1e-05)


class API_TestSplit3(unittest.TestCase):
    @test_with_pir_api
    def test_out(self):
        with base.program_guard(base.Program(), base.Program()):
            data = paddle.static.data('data', shape=[-1, 10], dtype='float64')
            x0, x1 = paddle.split(data, num_or_sections=(3, 7), axis=1)
            place = base.CPUPlace()
            exe = base.Executor(place)
            input1 = np.random.random([1, 10]).astype('float64')
            r0, r1 = exe.run(feed={"data": input1}, fetch_list=[x0, x1])
            ex_x0, ex_x1 = np.split(input1, (3,), axis=1)
            np.testing.assert_allclose(ex_x0, r0, rtol=1e-05)
            np.testing.assert_allclose(ex_x1, r1, rtol=1e-05)


class API_TestSplit4(unittest.TestCase):
    @test_with_pir_api
    def test_out(self):
        with base.program_guard(base.Program(), base.Program()):
            data = paddle.static.data('data', shape=[-1, 10], dtype='float64')
            index = paddle.static.data('index', shape=[1], dtype='int32')
            x0, x1 = paddle.split(data, num_or_sections=(3, index), axis=1)
            place = base.CPUPlace()
            exe = base.Executor(place)
            input1 = np.random.random([1, 10]).astype('float64')
            input2 = np.array([7]).astype('int32')
            r0, r1 = exe.run(
                feed={"data": input1, "index": input2}, fetch_list=[x0, x1]
            )
            ex_x0, ex_x1 = np.split(input1, (3,), axis=1)
            np.testing.assert_allclose(ex_x0, r0, rtol=1e-05)
            np.testing.assert_allclose(ex_x1, r1, rtol=1e-05)


class API_TestSplit5(unittest.TestCase):
    @test_with_pir_api
    def test_out(self):
        for use_cuda in (
            [False, True] if core.is_compiled_with_cuda() else [False]
        ):
            place = paddle.CUDAPlace(0) if use_cuda else paddle.CPUPlace()
            with base.program_guard(base.Program(), base.Program()):
                input_1 = np.random.random([5, 4]).astype("int32")
                # input is a variable which shape is [5, 4]
                input = paddle.to_tensor(input_1)
                n = paddle.full([1], 5, dtype='int32')
                out = paddle.split(input, [n])
                exe = paddle.static.Executor(place=place)
                re = exe.run(fetch_list=[out])
                re = re[0]
                ex_out = np.split(input_1, [5])
                ex_out = ex_out[0]
                np.testing.assert_allclose(ex_out, re, rtol=1e-05)


class API_TestSplit6(unittest.TestCase):
    @test_with_pir_api
    def test_out(self):
        with base.program_guard(base.Program(), base.Program()):
            data = paddle.static.data('data', shape=[-1, 10], dtype='float64')
            x0, x1 = paddle.split(data, num_or_sections=[1, 1], axis=0)
            place = base.CPUPlace()
            exe = base.Executor(place)
            input1 = np.random.random([2, 10]).astype('float64')
            r0, r1 = exe.run(feed={"data": input1}, fetch_list=[x0, x1])
            ex_x0, ex_x1 = np.split(input1, (1,), axis=0)
            np.testing.assert_allclose(ex_x0, r0, rtol=1e-05)
            np.testing.assert_allclose(ex_x1, r1, rtol=1e-05)


class API_TestDygraphFluidSplit(unittest.TestCase):
    def test_out1(self):
        with base.dygraph.guard():
            input_1 = np.random.random([4, 6, 6]).astype("int32")
            # input is a variable which shape is [4, 6, 6]
            input = paddle.to_tensor(input_1)
            x0, x1, x2 = paddle.split(input, num_or_sections=3, axis=1)
            x0_out = x0.numpy()
            x1_out = x1.numpy()
            x2_out = x2.numpy()
            ex_x0, ex_x1, ex_x2 = np.split(input_1, 3, axis=1)
            # input is a variable which shape is [4, 6, 6]
            input = paddle.to_tensor(input_1)
            input.stop_gradient = False
            x0, x1, x2 = paddle.split(input, num_or_sections=3, axis=1)
            eager_x0_out = x0.numpy()
            eager_x1_out = x1.numpy()
            eager_x2_out = x2.numpy()
            loss = x0.sum()
            loss.backward()
            manul_grad = np.zeros_like(input_1)
            manul_grad[:, :2, :] = 1
            np.testing.assert_allclose(input.gradient(), manul_grad, rtol=1e-05)
            np.testing.assert_allclose(ex_x0, eager_x0_out, rtol=1e-05)
            np.testing.assert_allclose(ex_x1, eager_x1_out, rtol=1e-05)
            np.testing.assert_allclose(ex_x2, eager_x2_out, rtol=1e-05)

        np.testing.assert_allclose(ex_x0, x0_out, rtol=1e-05)
        np.testing.assert_allclose(ex_x1, x1_out, rtol=1e-05)
        np.testing.assert_allclose(ex_x2, x2_out, rtol=1e-05)

    def test_out2(self):
        with base.dygraph.guard():
            input_1 = np.random.random([4, 6, 6]).astype("int32")
            # input is a variable which shape is [4, 6, 6]
            input = paddle.to_tensor(input_1)
            x0, x1, x2 = paddle.split(input, [2, 2, 2], axis=1)
            x0_out = x0.numpy()
            x1_out = x1.numpy()
            x2_out = x2.numpy()
            ex_x0, ex_x1, ex_x2 = np.split(input_1, 3, axis=1)
            # input is a variable which shape is [4, 6, 6]
            input = paddle.to_tensor(input_1)
            input.stop_gradient = False
            x0, x1, x2 = paddle.split(input, [2, 2, 2], axis=1)
            eager_x0_out = x0.numpy()
            eager_x1_out = x1.numpy()
            eager_x2_out = x2.numpy()
            loss = x0.sum()
            loss.backward()
            manul_grad = np.zeros_like(input_1)
            manul_grad[:, :2, :] = 1
            np.testing.assert_allclose(input.gradient(), manul_grad, rtol=1e-05)
            np.testing.assert_allclose(ex_x0, eager_x0_out, rtol=1e-05)
            np.testing.assert_allclose(ex_x1, eager_x1_out, rtol=1e-05)
            np.testing.assert_allclose(ex_x2, eager_x2_out, rtol=1e-05)

        np.testing.assert_allclose(ex_x0, x0_out, rtol=1e-05)
        np.testing.assert_allclose(ex_x1, x1_out, rtol=1e-05)
        np.testing.assert_allclose(ex_x2, x2_out, rtol=1e-05)


class API_TestDygraphSplit(unittest.TestCase):
    def test_out1(self):
        with base.dygraph.guard():
            input_1 = np.random.random([4, 6, 6]).astype("int32")
            # input is a variable which shape is [4, 6, 6]
            input = paddle.to_tensor(input_1)
            x0, x1, x2 = paddle.split(input, num_or_sections=3, axis=1)
            x0_out = x0.numpy()
            x1_out = x1.numpy()
            x2_out = x2.numpy()
            ex_x0, ex_x1, ex_x2 = np.split(input_1, 3, axis=1)

            # input is a variable which shape is [4, 6, 6]
            input = paddle.to_tensor(input_1)
            input.stop_gradient = False
            x0, x1, x2 = paddle.split(input, num_or_sections=3, axis=1)
            eager_x0_out = x0.numpy()
            eager_x1_out = x1.numpy()
            eager_x2_out = x2.numpy()
            loss = x0.sum()
            loss.backward()
            manul_grad = np.zeros_like(input_1)
            manul_grad[:, :2, :] = 1
            np.testing.assert_allclose(input.gradient(), manul_grad, rtol=1e-05)
            np.testing.assert_allclose(ex_x0, eager_x0_out, rtol=1e-05)
            np.testing.assert_allclose(ex_x1, eager_x1_out, rtol=1e-05)
            np.testing.assert_allclose(ex_x2, eager_x2_out, rtol=1e-05)

        np.testing.assert_allclose(ex_x0, x0_out, rtol=1e-05)
        np.testing.assert_allclose(ex_x1, x1_out, rtol=1e-05)
        np.testing.assert_allclose(ex_x2, x2_out, rtol=1e-05)

    def test_out2(self):
        with base.dygraph.guard():
            input_1 = np.random.random([4, 6, 6]).astype("bool")
            # input is a variable which shape is [4, 6, 6]
            input = paddle.to_tensor(input_1)
            x0, x1, x2 = paddle.split(input, num_or_sections=3, axis=1)
            x0_out = x0.numpy()
            x1_out = x1.numpy()
            x2_out = x2.numpy()
            ex_x0, ex_x1, ex_x2 = np.split(input_1, 3, axis=1)
        np.testing.assert_allclose(ex_x0, x0_out, rtol=1e-05)
        np.testing.assert_allclose(ex_x1, x1_out, rtol=1e-05)
        np.testing.assert_allclose(ex_x2, x2_out, rtol=1e-05)

    def test_out3(self):
        with base.dygraph.guard():
            np.random.seed(2021)
            input_1 = np.random.random([4, 6, 6]).astype("int32")
            # input is a variable which shape is [4, 6, 6]
            input = paddle.to_tensor(input_1)
            out_dy = paddle.split(input, [6], axis=1)
            out_dy = out_dy[0]
            out_dy_np = out_dy.numpy()
            ex_out = np.split(input_1, [6], axis=1)
            ex_out = ex_out[0]
            input = paddle.to_tensor(input_1)
            out_eager = paddle.split(input, [6], axis=1)
            out_eager = out_eager[0]
            out_eager_np = out_dy.numpy()
            np.testing.assert_allclose(ex_out, out_eager_np, rtol=1e-05)
        np.testing.assert_allclose(ex_out, out_dy_np, rtol=1e-05)

    def test_out_tensor_input(self):
        with base.dygraph.guard():
            input_1 = np.random.random([4, 6, 6]).astype("int32")
            # input is a variable which shape is [4, 6, 6]
            input = paddle.to_tensor(input_1)
            num1 = paddle.full(shape=[1], fill_value=2, dtype='int32')
            x0, x1, x2 = paddle.split(
                input, num_or_sections=[num1, 2, 2], axis=1
            )
            x0_out = x0.numpy()
            x1_out = x1.numpy()
            x2_out = x2.numpy()
            ex_x0, ex_x1, ex_x2 = np.split(input_1, 3, axis=1)
        np.testing.assert_allclose(ex_x0, x0_out, rtol=1e-05)
        np.testing.assert_allclose(ex_x1, x1_out, rtol=1e-05)
        np.testing.assert_allclose(ex_x2, x2_out, rtol=1e-05)

    def test_axis_tensor_input(self):
        with base.dygraph.guard():
            input_1 = np.random.random([4, 6, 6]).astype("int32")
            # input is a variable which shape is [4, 6, 6]
            input = paddle.to_tensor(input_1)
            num1 = paddle.full(shape=[1], fill_value=1, dtype='int32')
            x0, x1, x2 = paddle.split(
                input, num_or_sections=[2, 2, 2], axis=num1
            )
            x0_out = x0.numpy()
            x1_out = x1.numpy()
            x2_out = x2.numpy()
            ex_x0, ex_x1, ex_x2 = np.split(input_1, 3, axis=1)
        np.testing.assert_allclose(ex_x0, x0_out, rtol=1e-05)
        np.testing.assert_allclose(ex_x1, x1_out, rtol=1e-05)
        np.testing.assert_allclose(ex_x2, x2_out, rtol=1e-05)

    def test_negative_one_section(self):
        with base.dygraph.guard():
            input_1 = np.random.random([4, 6, 6]).astype("int32")
            # input is a variable which shape is [4, 6, 6]
            input = paddle.to_tensor(input_1)
            num1 = paddle.full(shape=[1], fill_value=1, dtype='int32')
            x0 = paddle.split(input, num_or_sections=[-1], axis=num1)
            x0_out = x0[0].numpy()
        np.testing.assert_array_equal(x0_out, input.numpy())


class API_TestEmptySplit(unittest.TestCase):
    def test_axis_input_empty_section(self):
        with base.dygraph.guard():
            input_1 = np.random.random([8, 6, 6]).astype("float32")
            # input is a variable which shape is [8, 6, 6]
            input = paddle.to_tensor(input_1)
            x0, x1, x2 = paddle.split(input, num_or_sections=[5, 0, 3])
            x0_out = x0.numpy()
            x1_out = x1.numpy()
            x2_out = x2.numpy()
            ex_x0, ex_x1, ex_x2 = np.split(
                input_1,
                [
                    5,
                    5,
                ],
            )
        np.testing.assert_allclose(ex_x0, x0_out, rtol=1e-05)
        np.testing.assert_allclose(ex_x1, x1_out, rtol=1e-05)
        np.testing.assert_allclose(ex_x2, x2_out, rtol=1e-05)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
