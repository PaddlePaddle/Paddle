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
import sys

sys.path.append("..")
import op_test
import numpy as np
import unittest
import paddle
import paddle.fluid.core as core
from paddle.fluid.op import Operator
import paddle.fluid as fluid
from paddle.fluid import compiler, Program, program_guard
from paddle.fluid.backward import append_backward
'''
class TestAssignOp(op_test.OpTest):
    def setUp(self):
        self.op_type = "assign"
        x = np.random.random(size=(100, 10)).astype('float32')
        self.inputs = {'X': x}
        self.outputs = {'Out': x}

    def test_forward(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place)

    def test_backward(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['X'], 'Out')


class TestAssignOpWithLoDTensorArray(unittest.TestCase):
    def test_assign_LoDTensorArray(self):
        main_program = Program()
        startup_program = Program()
        with program_guard(main_program):
            x = fluid.data(name='x', shape=[100, 10], dtype='float32')
            x.stop_gradient = False
            y = fluid.layers.fill_constant(
                shape=[100, 10], dtype='float32', value=1)
            z = fluid.layers.elementwise_add(x=x, y=y)
            i = fluid.layers.fill_constant(shape=[1], dtype='int64', value=0)
            init_array = fluid.layers.array_write(x=z, i=i)
            array = fluid.layers.assign(init_array)
            sums = fluid.layers.array_read(array=init_array, i=i)
            mean = fluid.layers.mean(sums)
            append_backward(mean)

        place = fluid.CUDAPlace(0) if core.is_compiled_with_cuda(
        ) else fluid.CPUPlace()
        exe = fluid.Executor(place)
        feed_x = np.random.random(size=(100, 10)).astype('float32')
        ones = np.ones((100, 10)).astype('float32')
        feed_add = feed_x + ones
        res = exe.run(main_program,
                      feed={'x': feed_x},
                      fetch_list=[sums.name, x.grad_name])
        self.assertTrue(np.allclose(res[0], feed_add))
        self.assertTrue(np.allclose(res[1], ones / 1000.0))


class TestAssignOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The type of input must be Variable or numpy.ndarray.
            x1 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.XPUPlace(0))
            self.assertRaises(TypeError, fluid.layers.assign, x1)
            x2 = np.array([[2.5, 2.5]], dtype='uint8')
            self.assertRaises(TypeError, fluid.layers.assign, x2)
'''

if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
