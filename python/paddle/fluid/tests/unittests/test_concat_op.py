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
from op_test import OpTest, skip_check_grad_ci
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard, core


class TestConcatOp(OpTest):
    def setUp(self):
        self.op_type = "concat"
        self.dtype = self.get_dtype()
        self.init_test_data()
        self.inputs = {'X': [('x0', self.x0), ('x1', self.x1), ('x2', self.x2)]}
        self.attrs = {'axis': self.axis}
        if self.axis < 0:
            self.actual_axis = self.axis + len(self.x0.shape)
            self.actual_axis = self.actual_axis if self.actual_axis > 0 else 0
        else:
            self.actual_axis = self.axis

        self.outputs = {
            'Out': np.concatenate(
                (self.x0, self.x1, self.x2), axis=self.actual_axis)
        }

    def get_dtype(self):
        return "float64"

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['x0'], 'Out')
        self.check_grad(['x1'], 'Out')
        self.check_grad(['x2'], 'Out')

    def init_test_data(self):
        self.x0 = np.random.random((5, 1, 4, 5)).astype(self.dtype)
        self.x1 = np.random.random((5, 2, 4, 5)).astype(self.dtype)
        self.x2 = np.random.random((5, 3, 4, 5)).astype(self.dtype)
        self.axis = 1


class TestConcatOp2(TestConcatOp):
    def init_test_data(self):
        self.x0 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.x1 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.x2 = np.random.random((2, 3, 4, 5)).astype(self.dtype)
        self.axis = 1


@skip_check_grad_ci(
    reason="The function 'check_grad' for large inputs is too slow.")
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


def create_test_AxisTensor(parent):
    class TestConcatAxisTensor(parent):
        def setUp(self):
            self.op_type = "concat"
            self.dtype = self.get_dtype()
            self.init_test_data()

            self.inputs = {
                'X': [('x0', self.x0), ('x1', self.x1), ('x2', self.x2)],
                'AxisTensor': np.array([self.axis]).astype("int32")
            }
            self.attrs = {}

            if self.axis < 0:
                self.actual_axis = self.axis + len(self.x0.shape)
                self.actual_axis = self.actual_axis if self.actual_axis > 0 else 0
            else:
                self.actual_axis = self.axis

            self.outputs = {
                'Out': np.concatenate(
                    (self.x0, self.x1, self.x2), axis=self.actual_axis)
            }

    cls_name = "{0}_{1}".format(parent.__name__, "AxisTensor")
    TestConcatAxisTensor.__name__ = cls_name
    globals()[cls_name] = TestConcatAxisTensor


create_test_AxisTensor(TestConcatOp)
create_test_AxisTensor(TestConcatOp2)
create_test_AxisTensor(TestConcatOp3)
create_test_AxisTensor(TestConcatOp4)
create_test_AxisTensor(TestConcatOp5)

#----------------Concat Fp16----------------


def create_test_fp16(parent):
    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestConcatFp16(parent):
        def get_dtype(self):
            return np.float16

    cls_name = "{0}_{1}".format(parent.__name__, "Fp16")
    TestConcatFp16.__name__ = cls_name
    globals()[cls_name] = TestConcatFp16


create_test_fp16(TestConcatOp)
create_test_fp16(TestConcatOp2)
create_test_fp16(TestConcatOp3)
create_test_fp16(TestConcatOp4)
create_test_fp16(TestConcatOp5)


class TestConcatOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of concat_op should be list.
            x1 = fluid.layers.data(shape=[4], dtype='int32', name='x1')
            fluid.layers.concat(x1)
            # The item in input must be Variable.
            x2 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CPUPlace())
            x3 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CPUPlace())
            self.assertRaises(TypeError, fluid.layers.concat, [x2])
            # The input dtype of concat_op must be float16(only support on GPU), float32, float64, int32, int64.
            x4 = fluid.layers.data(shape=[4], dtype='uint8', name='x4')
            x5 = fluid.layers.data(shape=[4], dtype='uint8', name='x5')
            self.assertRaises(TypeError, fluid.layers.concat, [x4, x5])
            x6 = fluid.layers.data(shape=[4], dtype='float16', name='x6')
            x7 = fluid.layers.data(shape=[4], dtype='float16', name='x7')
            fluid.layers.concat([x6, x7])

            # The type of axis in concat_op should be int or Variable.
            def test_axis_type():
                fluid.layers.concat([x6, x7], 3.2)

            self.assertRaises(TypeError, test_axis_type)


class TestConcatAPI(unittest.TestCase):
    def test_api(self):
        x_1 = fluid.data(shape=[None, 1, 4, 5], dtype='int32', name='x_1')
        fluid.layers.concat([x_1, x_1], 0)

        input_2 = np.random.random([2, 1, 4, 5]).astype("int32")
        input_3 = np.random.random([2, 2, 4, 5]).astype("int32")
        x_2 = fluid.data(shape=[2, 1, 4, 5], dtype='int32', name='x_2')
        x_3 = fluid.data(shape=[2, 2, 4, 5], dtype='int32', name='x_3')
        positive_1_int32 = fluid.layers.fill_constant([1], "int32", 1)
        positive_1_int64 = fluid.layers.fill_constant([1], "int64", 1)
        out_1 = fluid.layers.concat(input=[x_2, x_3], axis=1)
        out_2 = fluid.layers.concat(input=[x_2, x_3], axis=positive_1_int32)
        out_3 = fluid.layers.concat(input=[x_2, x_3], axis=positive_1_int64)

        exe = fluid.Executor(place=fluid.CPUPlace())
        [res_1, res_2, res_3] = exe.run(
            fluid.default_main_program(),
            feed={"x_1": input_2,
                  "x_2": input_2,
                  "x_3": input_3},
            fetch_list=[out_1, out_2, out_3])
        assert np.array_equal(res_1, np.concatenate((input_2, input_3), axis=1))
        assert np.array_equal(res_2, np.concatenate((input_2, input_3), axis=1))
        assert np.array_equal(res_3, np.concatenate((input_2, input_3), axis=1))


if __name__ == '__main__':
    unittest.main()
