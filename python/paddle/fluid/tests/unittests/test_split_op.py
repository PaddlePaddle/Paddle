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

from __future__ import print_function
import paddle
import unittest
import numpy as np
from op_test import OpTest, convert_float_to_uint16
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard, core
from paddle.fluid.framework import _test_eager_guard


class TestSplitOp(OpTest):
    def setUp(self):
        self._set_op_type()
        self.dtype = self.get_dtype()
        axis = 1
        if self.dtype == np.uint16:
            x = np.random.random((4, 5, 6)).astype(np.float32)
            out = np.split(x, [2, 3], axis)
            self.inputs = {'X': convert_float_to_uint16(x)}
            self.outputs = {'Out': [('out%d' % i, convert_float_to_uint16(out[i])) \
                for i in range(len(out))]}
        else:
            x = np.random.random((4, 5, 6)).astype(self.dtype)
            out = np.split(x, [2, 3], axis)
            self.inputs = {'X': x}
            self.outputs = {'Out': [('out%d' % i, out[i]) \
                for i in range(len(out))]}
        self.attrs = {'axis': axis, 'sections': [2, 1, 2]}

    def get_dtype(self):
        return "float64"

    def _set_op_type(self):
        self.op_type = "split"

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], ['out0', 'out1', 'out2'])


# test with attr(num)
class TestSplitOp_2(OpTest):
    def setUp(self):
        self._set_op_type()
        self.dtype = self.get_dtype()
        self.init_data()
        self.inputs = {'X': self.x}
        self.attrs = {
            'axis': self.axis,
            'sections': self.sections,
            'num': self.num
        }

        out = np.split(self.x, self.indices_or_sections, self.axis)
        self.outputs = {'Out': [('out%d' % i, out[i]) \
                                for i in range(len(out))]}

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
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], ['out0', 'out1', 'out2'])


# attr(axis) is Tensor
class TestSplitOp_AxisTensor(OpTest):
    def setUp(self):
        self._set_op_type()
        self.dtype = self.get_dtype()
        self.init_data()
        self.inputs = {
            'X': self.x,
            'AxisTensor': np.array([self.axis]).astype("int32")
        }
        self.attrs = {'sections': self.sections, 'num': self.num}

        out = np.split(self.x, self.indices_or_sections, self.axis)
        self.outputs = {'Out': [('out%d' % i, out[i]) \
                                for i in range(len(out))]}

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
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], ['out0', 'out1', 'out2'])


# attr(sections) is list containing Tensor
class TestSplitOp_SectionsTensor(OpTest):
    def setUp(self):
        self._set_op_type()
        self.dtype = self.get_dtype()
        self.init_data()
        self.inputs = {'X': self.x}

        sections_tensor = []
        for index, ele in enumerate(self.sections):
            sections_tensor.append(("x" + str(index), np.ones(
                (1)).astype('int32') * ele))

        self.inputs['SectionsTensorList'] = sections_tensor

        self.attrs = {
            'axis': self.axis,
            'sections': self.sections_infer,
            'num': self.num
        }

        out = np.split(self.x, self.indices_or_sections, self.axis)
        self.outputs = {'Out': [('out%d' % i, out[i]) \
                                for i in range(len(out))]}

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
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], ['out0', 'out1', 'out2'])


class TestSplitOp_unk_section(OpTest):
    def setUp(self):
        self._set_op_type()
        self.dtype = self.get_dtype()
        self.init_data()
        self.inputs = {'X': self.x}
        self.attrs = {
            'axis': self.axis,
            'sections': self.sections,
            'num': self.num
        }

        out = np.split(self.x, self.indices_or_sections, self.axis)
        self.outputs = {'Out': [('out%d' % i, out[i]) \
                                for i in range(len(out))]}

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
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], ['out0', 'out1', 'out2'])


class TestSplitByrefOp(OpTest):
    def _set_op_type(self):
        self.op_type = "split_byref"


#----------------Split Fp16----------------


def create_test_fp16(parent):
    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestSplitFp16(parent):
        def get_dtype(self):
            return np.float16

        def test_check_grad(self):
            pass

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16")
    TestSplitFp16.__name__ = cls_name
    globals()[cls_name] = TestSplitFp16


create_test_fp16(TestSplitOp)

#----------------Split Bf16----------------


def create_test_bf16(parent):
    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestSplitBf16(parent):
        def get_dtype(self):
            return np.uint16

        def test_check_output(self):
            place = core.CUDAPlace(0)
            self.check_output_with_place(place)

        def test_check_grad(self):
            pass

    cls_name = "{0}_{1}".format(parent.__name__, "Bf16")
    TestSplitBf16.__name__ = cls_name
    globals()[cls_name] = TestSplitBf16


create_test_bf16(TestSplitOp)


class TestSplitAPI(unittest.TestCase):
    def test_api(self):
        input_1 = np.random.random([4, 5, 6]).astype("int32")
        positive_1_int32 = fluid.layers.fill_constant([1], "int32", 1)
        positive_1_int64 = fluid.layers.fill_constant([1], "int64", 1)
        positive_2_int64 = fluid.layers.fill_constant([1], "int64", 2)
        x_1 = fluid.data(shape=[4, 5, 6], dtype='int32', name='x_1')
        x_2 = fluid.data(shape=[4, 5, None], dtype='int32', name='x_2')

        out_0, out_1, out_2 = fluid.layers.split(
            input=x_1,
            num_or_sections=[positive_2_int64, positive_1_int32, -1],
            dim=positive_1_int64)

        out_3, out_4, out_5 = fluid.layers.split(
            input=x_1, num_or_sections=[2, 1, 2], dim=positive_1_int32)
        fluid.layers.split(input=x_2, num_or_sections=2, dim=2)

        exe = fluid.Executor(place=fluid.CPUPlace())
        [res_0, res_1, res_2, res_3, res_4, res_5] = exe.run(
            fluid.default_main_program(),
            feed={"x_1": input_1,
                  "x_2": input_1},
            fetch_list=[out_0, out_1, out_2, out_3, out_4, out_5])

        out = np.split(input_1, [2, 3], 1)
        assert np.array_equal(res_0, out[0])
        assert np.array_equal(res_1, out[1])
        assert np.array_equal(res_2, out[2])
        assert np.array_equal(res_3, out[0])
        assert np.array_equal(res_4, out[1])
        assert np.array_equal(res_5, out[2])


class TestSplitOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The type of axis in split_op should be int or Variable.
            def test_axis_type():
                x6 = fluid.layers.data(shape=[4], dtype='float16', name='x3')
                fluid.layers.split(input=x6, num_or_sections=2, dim=3.2)

            self.assertRaises(TypeError, test_axis_type)

            # The type of axis in split_op should be int or Variable.
            def test_axis_variable_type():
                x9 = fluid.layers.data(shape=[4], dtype='float16', name='x9')
                x10 = fluid.layers.data(shape=[1], dtype='float16', name='x10')
                fluid.layers.split(input=x9, num_or_sections=2, dim=x10)

            self.assertRaises(TypeError, test_axis_variable_type)

            # The type of num_or_sections in split_op should be int, tuple or list.
            def test_num_or_sections_type():
                x6 = fluid.layers.data(shape=[4], dtype='float16', name='x4')
                fluid.layers.split(input=x6, num_or_sections=2.1, dim=3)

            self.assertRaises(TypeError, test_num_or_sections_type)

            def test_num_or_sections_type_tensor():
                x7 = fluid.layers.data(shape=[4], dtype='float16', name='x5')
                paddle.split(input=x7, num_or_sections=2.1, dim=3)

            self.assertRaises(TypeError, test_num_or_sections_type_tensor)

            def test_axis_type_tensor():
                x8 = fluid.layers.data(shape=[4], dtype='float16', name='x6')
                paddle.split(input=x8, num_or_sections=2, dim=3.2)

            self.assertRaises(TypeError, test_axis_type_tensor)


class API_TestSplit(unittest.TestCase):
    def test_out(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            data1 = fluid.layers.data('data1', shape=[4, 6, 6], dtype='float64')
            data2 = fluid.layers.data('data2', shape=[1], dtype='int32')
            x0, x1, x2 = paddle.split(data1, num_or_sections=3, axis=data2)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            input1 = np.random.random([4, 6, 6]).astype('float64')
            input2 = np.array([2]).astype('int32')
            r0, r1, r2, = exe.run(feed={"data1": input1,
                                        "data2": input2},
                                  fetch_list=[x0, x1, x2])
            ex_x0, ex_x1, ex_x2 = np.split(input1, 3, axis=2)
            self.assertTrue(np.allclose(ex_x0, r0))
            self.assertTrue(np.allclose(ex_x1, r1))
            self.assertTrue(np.allclose(ex_x2, r2))


class API_TestSplit2(unittest.TestCase):
    def test_out(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            data1 = fluid.layers.data('data1', shape=[4, 6, 6], dtype='float64')
            x0, x1, x2 = paddle.split(data1, num_or_sections=3, axis=2)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            input1 = np.random.random([4, 6, 6]).astype('float64')
            r0, r1, r2, = exe.run(feed={"data1": input1},
                                  fetch_list=[x0, x1, x2])
            ex_x0, ex_x1, ex_x2 = np.split(input1, 3, axis=2)
            self.assertTrue(np.allclose(ex_x0, r0))
            self.assertTrue(np.allclose(ex_x1, r1))
            self.assertTrue(np.allclose(ex_x2, r2))


class API_TestSplit3(unittest.TestCase):
    def test_out(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            data = fluid.layers.data('data', shape=[-1, 10], dtype='float64')
            x0, x1 = paddle.split(data, num_or_sections=(3, 7), axis=1)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            input1 = np.random.random([1, 10]).astype('float64')
            r0, r1 = exe.run(feed={"data": input1}, fetch_list=[x0, x1])
            ex_x0, ex_x1 = np.split(input1, (3, ), axis=1)
            self.assertTrue(np.allclose(ex_x0, r0))
            self.assertTrue(np.allclose(ex_x1, r1))


class API_TestSplit4(unittest.TestCase):
    def test_out(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            data = fluid.layers.data('data', shape=[-1, 10], dtype='float64')
            index = fluid.layers.data('index', shape=[1], dtype='int32')
            x0, x1 = paddle.split(data, num_or_sections=(3, index), axis=1)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            input1 = np.random.random([1, 10]).astype('float64')
            input2 = np.array([7]).astype('int32')
            r0, r1 = exe.run(feed={"data": input1,
                                   "index": input2},
                             fetch_list=[x0, x1])
            ex_x0, ex_x1 = np.split(input1, (3, ), axis=1)
            self.assertTrue(np.allclose(ex_x0, r0))
            self.assertTrue(np.allclose(ex_x1, r1))


class API_TestDygraphSplit(unittest.TestCase):
    def test_out1(self):
        with fluid.dygraph.guard():
            input_1 = np.random.random([4, 6, 6]).astype("int32")
            # input is a variable which shape is [4, 6, 6]
            input = paddle.to_tensor(input_1)
            x0, x1, x2 = paddle.split(input, num_or_sections=3, axis=1)
            x0_out = x0.numpy()
            x1_out = x1.numpy()
            x2_out = x2.numpy()
            ex_x0, ex_x1, ex_x2 = np.split(input_1, 3, axis=1)

            with _test_eager_guard():
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
                self.assertTrue(np.allclose(input.gradient(), manul_grad))
                self.assertTrue(np.allclose(ex_x0, eager_x0_out))
                self.assertTrue(np.allclose(ex_x1, eager_x1_out))
                self.assertTrue(np.allclose(ex_x2, eager_x2_out))

        self.assertTrue(np.allclose(ex_x0, x0_out))
        self.assertTrue(np.allclose(ex_x1, x1_out))
        self.assertTrue(np.allclose(ex_x2, x2_out))

    def test_out2(self):
        with fluid.dygraph.guard():
            input_1 = np.random.random([4, 6, 6]).astype("bool")
            # input is a variable which shape is [4, 6, 6]
            input = paddle.to_tensor(input_1)
            x0, x1, x2 = paddle.split(input, num_or_sections=3, axis=1)
            x0_out = x0.numpy()
            x1_out = x1.numpy()
            x2_out = x2.numpy()
            ex_x0, ex_x1, ex_x2 = np.split(input_1, 3, axis=1)
        self.assertTrue(np.allclose(ex_x0, x0_out))
        self.assertTrue(np.allclose(ex_x1, x1_out))
        self.assertTrue(np.allclose(ex_x2, x2_out))

    def test_out_tensor_input(self):
        with fluid.dygraph.guard():
            input_1 = np.random.random([4, 6, 6]).astype("int32")
            # input is a variable which shape is [4, 6, 6]
            input = paddle.to_tensor(input_1)
            num1 = paddle.full(shape=[1], fill_value=2, dtype='int32')
            x0, x1, x2 = paddle.split(
                input, num_or_sections=[num1, 2, 2], axis=1)
            x0_out = x0.numpy()
            x1_out = x1.numpy()
            x2_out = x2.numpy()
            ex_x0, ex_x1, ex_x2 = np.split(input_1, 3, axis=1)
        self.assertTrue(np.allclose(ex_x0, x0_out))
        self.assertTrue(np.allclose(ex_x1, x1_out))
        self.assertTrue(np.allclose(ex_x2, x2_out))

    def test_axis_tensor_input(self):
        with fluid.dygraph.guard():
            input_1 = np.random.random([4, 6, 6]).astype("int32")
            # input is a variable which shape is [4, 6, 6]
            input = paddle.to_tensor(input_1)
            num1 = paddle.full(shape=[1], fill_value=1, dtype='int32')
            x0, x1, x2 = paddle.split(
                input, num_or_sections=[2, 2, 2], axis=num1)
            x0_out = x0.numpy()
            x1_out = x1.numpy()
            x2_out = x2.numpy()
            ex_x0, ex_x1, ex_x2 = np.split(input_1, 3, axis=1)
        self.assertTrue(np.allclose(ex_x0, x0_out))
        self.assertTrue(np.allclose(ex_x1, x1_out))
        self.assertTrue(np.allclose(ex_x2, x2_out))


class API_TestEmptySplit(unittest.TestCase):
    def test_axis_input_empty_section(self):
        with fluid.dygraph.guard():
            input_1 = np.random.random([8, 6, 6]).astype("float32")
            # input is a variable which shape is [8, 6, 6]
            input = paddle.to_tensor(input_1)
            x0, x1, x2 = paddle.split(input, num_or_sections=[5, 0, 3])
            x0_out = x0.numpy()
            x1_out = x1.numpy()
            x2_out = x2.numpy()
            ex_x0, ex_x1, ex_x2 = np.split(input_1, [
                5,
                5,
            ])
        self.assertTrue(np.allclose(ex_x0, x0_out))
        self.assertTrue(np.allclose(ex_x1, x1_out))
        self.assertTrue(np.allclose(ex_x2, x2_out))


if __name__ == '__main__':
    unittest.main()
