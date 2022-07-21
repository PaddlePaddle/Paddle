# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
import paddle
import paddle.fluid as fluid
from paddle.fluid import core


class TestReverseOp(OpTest):

    def initTestCase(self):
        self.x = np.random.random((3, 40)).astype('float64')
        self.axis = [0]

    def setUp(self):
        self.initTestCase()
        self.op_type = "reverse"
        self.inputs = {"X": self.x}
        self.attrs = {'axis': self.axis}
        out = self.x
        for a in self.axis:
            out = np.flip(out, axis=a)
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(['X'], 'Out', check_eager=True)


class TestCase0(TestReverseOp):

    def initTestCase(self):
        self.x = np.random.random((3, 40)).astype('float64')
        self.axis = [1]


class TestCase0_neg(TestReverseOp):

    def initTestCase(self):
        self.x = np.random.random((3, 40)).astype('float64')
        self.axis = [-1]


class TestCase1(TestReverseOp):

    def initTestCase(self):
        self.x = np.random.random((3, 40)).astype('float64')
        self.axis = [0, 1]


class TestCase1_neg(TestReverseOp):

    def initTestCase(self):
        self.x = np.random.random((3, 40)).astype('float64')
        self.axis = [0, -1]


class TestCase2(TestReverseOp):

    def initTestCase(self):
        self.x = np.random.random((3, 4, 10)).astype('float64')
        self.axis = [0, 2]


class TestCase2_neg(TestReverseOp):

    def initTestCase(self):
        self.x = np.random.random((3, 4, 10)).astype('float64')
        self.axis = [0, -2]


class TestCase3(TestReverseOp):

    def initTestCase(self):
        self.x = np.random.random((3, 4, 10)).astype('float64')
        self.axis = [1, 2]


class TestCase3_neg(TestReverseOp):

    def initTestCase(self):
        self.x = np.random.random((3, 4, 10)).astype('float64')
        self.axis = [-1, -2]


class TestCase4(unittest.TestCase):

    def test_error(self):
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)

        train_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(train_program, startup_program):
            label = fluid.layers.data(name="label",
                                      shape=[1, 1, 1, 1, 1, 1, 1, 1],
                                      dtype="int64")
            rev = fluid.layers.reverse(label, axis=[-1, -2])

        def _run_program():
            x = np.random.random(size=(10, 1, 1, 1, 1, 1, 1)).astype('int64')
            exe.run(train_program, feed={"label": x})

        self.assertRaises(IndexError, _run_program)


class TestReverseLoDTensorArray(unittest.TestCase):

    def setUp(self):
        self.shapes = [[5, 25], [5, 20], [5, 5]]
        self.place = fluid.CUDAPlace(
            0) if fluid.is_compiled_with_cuda() else fluid.CPUPlace()
        self.exe = fluid.Executor(self.place)

    def run_program(self, arr_len, axis=0):
        main_program = fluid.Program()

        with fluid.program_guard(main_program):
            inputs, inputs_data = [], []
            for i in range(arr_len):
                x = fluid.data("x%s" % i, self.shapes[i], dtype='float32')
                x.stop_gradient = False
                inputs.append(x)
                inputs_data.append(
                    np.random.random(self.shapes[i]).astype('float32'))

            tensor_array = fluid.layers.create_array(dtype='float32')
            for i in range(arr_len):
                idx = fluid.layers.array_length(tensor_array)
                fluid.layers.array_write(inputs[i], idx, tensor_array)

            reverse_array = fluid.layers.reverse(tensor_array, axis=axis)
            output, _ = fluid.layers.tensor_array_to_tensor(reverse_array)
            loss = fluid.layers.reduce_sum(output)
            fluid.backward.append_backward(loss)
            input_grads = list(
                map(main_program.global_block().var,
                    [x.name + "@GRAD" for x in inputs]))

            feed_dict = dict(zip([x.name for x in inputs], inputs_data))
            res = self.exe.run(main_program,
                               feed=feed_dict,
                               fetch_list=input_grads + [output.name])

            return np.hstack(inputs_data[::-1]), res

    def test_case1(self):
        gt, res = self.run_program(arr_len=3)
        self.check_output(gt, res)
        # test with tuple type of axis
        gt, res = self.run_program(arr_len=3, axis=(0, ))
        self.check_output(gt, res)

    def test_case2(self):
        gt, res = self.run_program(arr_len=1)
        self.check_output(gt, res)
        # test with list type of axis
        gt, res = self.run_program(arr_len=1, axis=[0])
        self.check_output(gt, res)

    def check_output(self, gt, res):
        arr_len = len(res) - 1
        reversed_array = res[-1]
        # check output
        self.assertTrue(np.array_equal(gt, reversed_array))
        # check grad
        for i in range(arr_len):
            self.assertTrue(np.array_equal(res[i], np.ones_like(res[i])))

    def test_raise_error(self):
        # The len(axis) should be 1 is input(X) is LoDTensorArray
        with self.assertRaises(Exception):
            self.run_program(arr_len=3, axis=[0, 1])
        # The value of axis should be 0 is input(X) is LoDTensorArray
        with self.assertRaises(Exception):
            self.run_program(arr_len=3, axis=1)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
