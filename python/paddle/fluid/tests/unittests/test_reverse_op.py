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

import os
import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import core
import gradient_checker
from decorator_helper import prog_scope
import paddle.fluid.layers as layers

from paddle.fluid.framework import program_guard, Program
from test_attribute_var import UnittestBase


class TestReverseOp(OpTest):

    def initTestCase(self):
        self.x = np.random.random((3, 40)).astype('float64')
        self.axis = [0]

    def setUp(self):
        self.initTestCase()
        self.op_type = "reverse"
        self.python_api = fluid.layers.reverse
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
        np.testing.assert_array_equal(gt, reversed_array)
        # check grad
        for i in range(arr_len):
            np.testing.assert_array_equal(res[i], np.ones_like(res[i]))

    def test_raise_error(self):
        # The len(axis) should be 1 is input(X) is LoDTensorArray
        with self.assertRaises(Exception):
            self.run_program(arr_len=3, axis=[0, 1])
        # The value of axis should be 0 is input(X) is LoDTensorArray
        with self.assertRaises(Exception):
            self.run_program(arr_len=3, axis=1)


class TestReverseAxisTensor(UnittestBase):

    def init_info(self):
        self.shapes = [[2, 3, 4]]
        self.save_path = os.path.join(self.temp_dir.name, self.path_prefix())

    def test_static(self):
        main_prog = Program()
        starup_prog = Program()
        with program_guard(main_prog, starup_prog):
            fc = paddle.nn.Linear(4, 10)
            x = paddle.randn([2, 3, 4])
            x.stop_gradient = False
            feat = fc(x)  # [2,3,10]

            out = self.call_func(feat)

            sgd = paddle.optimizer.SGD()
            sgd.minimize(paddle.mean(out))
            self.assertTrue(self.var_prefix() in str(main_prog))

            exe = paddle.static.Executor()
            exe.run(starup_prog)
            res = exe.run(fetch_list=[feat, out])
            gt = res[0][::-1, :, ::-1]
            np.testing.assert_allclose(res[1], gt)

            paddle.static.save_inference_model(self.save_path, [x], [feat, out],
                                               exe)
            # Test for Inference Predictor
            infer_outs = self.infer_prog()
            gt = infer_outs[0][::-1, :, ::-1]
            np.testing.assert_allclose(infer_outs[1], gt)

    def path_prefix(self):
        return 'reverse_tensor'

    def var_prefix(self):
        return "Var["

    def call_func(self, x):
        # axes is a Variable
        axes = paddle.assign([0, 2])
        out = paddle.fluid.layers.reverse(x, axes)
        return out


class TestReverseAxisListTensor(TestReverseAxisTensor):

    def path_prefix(self):
        return 'reverse_tensors'

    def var_prefix(self):
        return "Vars["

    def call_func(self, x):
        # axes is a List[Variable]
        axes = [paddle.assign([0]), paddle.assign([2])]
        out = paddle.fluid.layers.reverse(x, axes)

        # check attrs
        axis_attrs = paddle.static.default_main_program().block(
            0).ops[-1].all_attrs()["axis"]
        self.assertTrue(axis_attrs[0].name, axes[0].name)
        self.assertTrue(axis_attrs[1].name, axes[1].name)
        return out


class TestReverseDoubleGradCheck(unittest.TestCase):

    def reverse_wrapper(self, x):
        return fluid.layers.reverse(x[0], [0, 1])

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        eps = 0.005
        dtype = np.float64

        data = layers.data('data', [3, 4], False, dtype)
        data.persistable = True
        out = fluid.layers.reverse(data, [0, 1])
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)

        gradient_checker.double_grad_check([data],
                                           out,
                                           x_init=[data_arr],
                                           place=place,
                                           eps=eps)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        gradient_checker.double_grad_check_for_dygraph(self.reverse_wrapper,
                                                       [data],
                                                       out,
                                                       x_init=[data_arr],
                                                       place=place)

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


class TestReverseTripleGradCheck(unittest.TestCase):

    def reverse_wrapper(self, x):
        return fluid.layers.reverse(x[0], [0, 1])

    @prog_scope()
    def func(self, place):
        # the shape of input variable should be clearly specified, not inlcude -1.
        eps = 0.005
        dtype = np.float32

        data = layers.data('data', [2, 3], False, dtype)
        data.persistable = True
        out = fluid.layers.reverse(data, [0, 1])
        data_arr = np.random.uniform(-1, 1, data.shape).astype(dtype)

        gradient_checker.triple_grad_check([data],
                                           out,
                                           x_init=[data_arr],
                                           place=place,
                                           eps=eps)
        fluid.set_flags({"FLAGS_retain_grad_for_all_tensor": True})
        gradient_checker.triple_grad_check_for_dygraph(self.reverse_wrapper,
                                                       [data],
                                                       out,
                                                       x_init=[data_arr],
                                                       place=place)

    def test_grad(self):
        paddle.enable_static()
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for p in places:
            self.func(p)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
