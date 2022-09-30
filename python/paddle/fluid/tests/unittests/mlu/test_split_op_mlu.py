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

import numpy as np
import unittest
import sys

sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core

paddle.enable_static()
SEED = 2021


class TestCase1(OpTest):

    def setUp(self):
        self.set_mlu()
        self.set_example()
        self.op_type = "split"
        self.place = paddle.device.MLUPlace(0)
        ipt = self.x.astype(self.dtype)
        axis = self.axis if isinstance(self.axis, int) else int(self.axis[0])
        tmp_outs = np.split(ipt,
                            axis=axis,
                            indices_or_sections=self.num_or_sections)
        tmp_outs = [o.astype(self.dtype) for o in tmp_outs]
        self.outputs = {'Out': []}
        self.outs = []
        for i, o in enumerate(tmp_outs):
            self.outputs["Out"].append((str(i), o))
            self.outs.append(str(i))

        self.attrs = {"axis": self.axis, "num": self.num_or_sections}
        self.inputs = {}
        self.inputs.update({'X': ipt.astype(self.dtype)})

    def set_mlu(self):
        self.__class__.use_mlu = True
        self.__class__.op_type = "split"

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def set_example(self):
        self.dtype = "float32"
        self.x = np.random.random((2, 4, 6))
        self.axis = 1
        self.num_or_sections = 2


class TestCase2(TestCase1):

    def set_example(self):
        self.dtype = "float32"
        self.x = np.random.random((20, 4, 50))
        self.axis = 0
        self.num_or_sections = 4


class TestCase4(TestCase1):

    def set_example(self):
        self.dtype = "float16"
        self.x = np.random.random((4, 50, 20))
        self.axis = 2
        self.num_or_sections = 4


# Test Sections
class TestCase5(TestCase1):

    def set_example(self):
        super().set_example()
        self.x = np.random.random((2, 10, 4))
        self.axis = 1
        self.num_or_sections = [2, 4, 8]

    def setUp(self):
        super().setUp()
        self.attrs.update({"sections": [2, 2, 4, 2], "num": 0})


class API_TestSplit(unittest.TestCase):

    def test_out(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            data = fluid.layers.data('data', shape=[-1, 10], dtype='float32')
            x0, x1 = paddle.split(data, num_or_sections=(3, 7), axis=1)
            place = fluid.MLUPlace(0)
            exe = fluid.Executor(place)
            input1 = np.random.random([1, 10]).astype('float32')
            r0, r1 = exe.run(feed={"data": input1}, fetch_list=[x0, x1])
            ex_x0, ex_x1 = np.split(input1, (3, ), axis=1)
            np.testing.assert_allclose(ex_x0, r0)
            np.testing.assert_allclose(ex_x1, r1)


class API_TestSplit2(unittest.TestCase):

    def test_out(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            data = fluid.layers.data('data', shape=[-1, 10], dtype='float32')
            x0, x1 = paddle.split(data, num_or_sections=2, axis=1)
            place = fluid.MLUPlace(0)
            exe = fluid.Executor(place)
            input1 = np.random.random([1, 10]).astype('float32')
            r0, r1 = exe.run(feed={"data": input1}, fetch_list=[x0, x1])
            ex_x0, ex_x1 = np.split(input1, 2, axis=1)
            np.testing.assert_allclose(ex_x0, r0)
            np.testing.assert_allclose(ex_x1, r1)


class API_TestDygraphSplit(unittest.TestCase):

    def test_out1(self):
        with fluid.dygraph.guard(paddle.MLUPlace(0)):
            input_1 = np.random.random([4, 6, 6]).astype("int32")
            # input is a variable which shape is [4, 6, 6]
            input = fluid.dygraph.to_variable(input_1)
            x0, x1, x2 = paddle.split(input, num_or_sections=3, axis=1)
            x0_out = x0.numpy()
            x1_out = x1.numpy()
            x2_out = x2.numpy()
            ex_x0, ex_x1, ex_x2 = np.split(input_1, 3, axis=1)
        np.testing.assert_allclose(ex_x0, x0_out)
        np.testing.assert_allclose(ex_x1, x1_out)
        np.testing.assert_allclose(ex_x2, x2_out)

    def test_out2(self):
        with fluid.dygraph.guard(paddle.MLUPlace(0)):
            input_1 = np.random.random([4, 6, 6]).astype("int32")
            # input is a variable which shape is [4, 6, 6]
            input = fluid.dygraph.to_variable(input_1)
            x0, x1, x2 = paddle.split(input, num_or_sections=[1, 2, 3], axis=1)
            x0_out = x0.numpy()
            x1_out = x1.numpy()
            x2_out = x2.numpy()
            ex_x0, ex_x1, ex_x2 = np.split(input_1, (1, 3), axis=1)
        np.testing.assert_allclose(ex_x0, x0_out)
        np.testing.assert_allclose(ex_x1, x1_out)
        np.testing.assert_allclose(ex_x2, x2_out)


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

        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True

    def init_data(self):
        self.x = np.random.random((4, 5, 6)).astype(self.dtype)
        self.axis = 2
        self.sections = []
        self.num = 3
        self.indices_or_sections = 3

    def get_dtype(self):
        return "float"

    def _set_op_type(self):
        self.op_type = "split"

    def test_check_output(self):
        self.check_output_with_place(self.place)


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

        self.place = paddle.device.MLUPlace(0)
        self.__class__.use_mlu = True

    def init_data(self):
        self.x = np.random.random((4, 5, 6)).astype(self.dtype)
        self.axis = 1
        self.sections = [2, 1, 2]
        self.sections_infer = [-1, -1, -1]
        self.num = 0
        self.indices_or_sections = [2, 3]

    def get_dtype(self):
        return "float"

    def _set_op_type(self):
        self.op_type = "split"

    def test_check_output(self):
        self.check_output_with_place(self.place)


if __name__ == '__main__':
    unittest.main()
