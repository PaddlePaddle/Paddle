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

import unittest
import numpy as np
from op_test import OpTest
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard, core


class TestSplitOp(OpTest):
    def setUp(self):
        self._set_op_type()
        self.dtype = self.get_dtype()
        axis = 1
        x = np.random.random((4, 5, 6)).astype(self.dtype)
        out = np.split(x, [2, 3], axis)
        self.inputs = {'X': x}
        self.attrs = {'axis': axis, 'sections': [2, 1, 2]}
        self.outputs = {'Out': [('out%d' % i, out[i]) \
            for i in range(len(out))]}

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

            # The type of num_or_sections in split_op should be int, tuple or list.
            def test_num_or_sections_type():
                x6 = fluid.layers.data(shape=[4], dtype='float16', name='x4')
                fluid.layers.split(input=x6, num_or_sections=2.1, dim=3)

            self.assertRaises(TypeError, test_num_or_sections_type)


if __name__ == '__main__':
    unittest.main()
