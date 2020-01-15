#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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


def np_partial_concat(inputs, start, length):
    assert (len(inputs[0].shape) == 2)
    size = inputs[0].shape[1]
    assert (start >= -size and start < size)

    if start < 0:
        start += size
    if length < 0:
        length = size - start
    assert (size >= start + length)

    elems = []
    for elem in inputs:
        assert (elem.shape == inputs[0].shape)
        elems.append(elem[:, start:start + length])
    return np.concatenate(elems, axis=1)


@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestPartialConcatOp(OpTest):
    def setUp(self):
        self.op_type = "partial_concat"
        self.dtype = self.get_dtype()
        self.init_test_data()
        self.init_attrs()
        self.inputs = {'X': [('x0', self.x0), ('x1', self.x1), ('x2', self.x2)]}
        self.outputs = {
            'Out': np_partial_concat((self.x0, self.x1, self.x2),
                                     self.start_index, self.length)
        }

    def get_dtype(self):
        return "float32"

    def init_attrs(self):
        self.start_index = 0
        self.length = -1

    def test_check_output(self):
        self.check_output()

    def init_test_data(self):
        self.x0 = np.random.random((10, 20)).astype(self.dtype)
        self.x1 = np.random.random((10, 20)).astype(self.dtype)
        self.x2 = np.random.random((10, 20)).astype(self.dtype)


@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestPartialConcatOp2(TestPartialConcatOp):
    def init_attrs(self):
        self.start_index = 5
        self.length = -1


@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestPartialConcatOp3(TestPartialConcatOp):
    def init_attrs(self):
        self.start_index = -5
        self.length = -1


@skip_check_grad_ci(reason="For inference, check_grad is not required.")
class TestPartialConcatOp4(TestPartialConcatOp):
    def init_attrs(self):
        self.start_index = 5
        self.length = 3


#class TestPartialConcatOpError(unittest.TestCase):
#    def test_errors(self):
#        with program_guard(Program(), Program()):
#            # The input type of concat_op should be list.
#            x1 = fluid.layers.data(shape=[4], dtype='int32', name='x1')
#            fluid.layers.concat(x1)
#            # The item in input must be Variable.
#            x2 = fluid.create_lod_tensor(
#                np.array([[-1]]), [[1]], fluid.CPUPlace())
#            x3 = fluid.create_lod_tensor(
#                np.array([[-1]]), [[1]], fluid.CPUPlace())
#            self.assertRaises(TypeError, fluid.layers.concat, [x2])
#            # The input dtype of concat_op must be float16(only support on GPU), float32, float64, int32, int64.
#            x4 = fluid.layers.data(shape=[4], dtype='uint8', name='x4')
#            x5 = fluid.layers.data(shape=[4], dtype='uint8', name='x5')
#            self.assertRaises(TypeError, fluid.layers.concat, [x4, x5])
#            x6 = fluid.layers.data(shape=[4], dtype='float16', name='x6')
#            x7 = fluid.layers.data(shape=[4], dtype='float16', name='x7')
#            fluid.layers.concat([x6, x7])
#
#            # The type of axis in concat_op should be int or Variable.
#            def test_axis_type():
#                fluid.layers.concat([x6, x7], 3.2)
#
#            self.assertRaises(TypeError, test_axis_type)


class TestPartialConcatAPI(unittest.TestCase):
    def test_api(self):
        x1 = fluid.data(shape=[10, 20], dtype='float32', name='x1')
        x2 = fluid.data(shape=[10, 20], dtype='float32', name='x2')
        out1 = fluid.layers.partial_concat([x1, x2], 1, 2)
        out2 = fluid.layers.partial_concat([x1, x2], 1, -1)
        out3 = fluid.layers.partial_concat([x1, x2], -5, -1)

        input1 = np.random.random([10, 20]).astype("float32")
        input2 = np.random.random([10, 20]).astype("float32")
        exe = fluid.Executor(place=fluid.CPUPlace())
        [res1, res2, res3] = exe.run(fluid.default_main_program(),
                                     feed={"x1": input1,
                                           "x2": input2},
                                     fetch_list=[out1, out2, out3])
        assert np.array_equal(res1, np_partial_concat((input1, input2), 1, 2))
        assert np.array_equal(res2, np_partial_concat((input1, input2), 1, -1))
        assert np.array_equal(res3, np_partial_concat((input1, input2), -5, -1))


if __name__ == '__main__':
    unittest.main()
