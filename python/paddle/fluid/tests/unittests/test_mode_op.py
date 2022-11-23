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
from op_test import OpTest
import paddle
import paddle.fluid as fluid


def _mode1D(a):
    sorted_inds = np.argsort(a, kind='stable')
    sorted_array = a[sorted_inds]
    max_freq = 0
    cur_freq = 0
    mode = -1
    for i in range(len(sorted_array)):
        cur_freq += 1
        if i == len(sorted_array) - 1 or sorted_array[i] != sorted_array[i + 1]:
            if cur_freq > max_freq:
                mode = sorted_array[i]
                index = sorted_inds[i]
                max_freq = cur_freq
        cur_freq = 0
    return mode, index


def cal_mode(a, axis, keepdim=False):
    if axis < 0:
        axis = len(a.shape) + axis
    in_dims = list(range(a.ndim))
    a_view = np.transpose(a, in_dims[:axis] + in_dims[axis + 1:] + [axis])
    inds = np.ndindex(a_view.shape[:-1])
    modes = np.empty(a_view.shape[:-1], dtype=a.dtype)
    indexes = np.empty(a_view.shape[:-1], dtype=np.int64)
    for ind in inds:
        modes[ind], indexes[ind] = _mode1D(a_view[ind])
    if keepdim:
        newshape = list(a.shape)
        newshape[axis] = 1
        modes = modes.reshape(newshape)
        indexes = indexes.reshape(newshape)
    return modes, indexes


class TestModeOp(OpTest):

    def init_args(self):
        self.axis = 1

    def setUp(self):
        self.op_type = "mode"
        self.python_api = paddle.mode
        self.dtype = np.float64
        np.random.seed(666)
        self.input_data = np.random.rand(2, 64, 1)
        self.init_args()
        self.inputs = {'X': self.input_data}
        self.attrs = {'axis': self.axis}
        output, indices = cal_mode(self.input_data, axis=self.axis)
        self.outputs = {'Out': output, 'Indices': indices}

    def test_check_output(self):
        paddle.enable_static()
        self.check_output(check_eager=True)

    def test_check_grad(self):
        paddle.enable_static()
        self.check_grad(set(['X']), 'Out', check_eager=True)


class TestModeOpLastdim(OpTest):

    def init_args(self):
        self.axis = -1

    def setUp(self):
        self.op_type = "mode"
        self.python_api = paddle.mode
        self.dtype = np.float64
        np.random.seed(666)
        self.input_data = np.random.rand(2, 1, 1, 2, 30)
        self.init_args()
        self.inputs = {'X': self.input_data}
        self.attrs = {'axis': self.axis}
        output, indices = cal_mode(self.input_data, axis=self.axis)
        self.outputs = {'Out': output, 'Indices': indices}

    def test_check_output(self):
        paddle.enable_static()
        self.check_output(check_eager=True)

    def test_check_grad(self):
        paddle.enable_static()
        self.check_grad(set(['X']), 'Out', check_eager=True)


class TestModeOpKernels(unittest.TestCase):

    def setUp(self):
        self.axises = [-1, 1]
        np.random.seed(666)
        self.inputs = np.ceil(np.random.rand(2, 10, 10) * 1000)

    def test_mode_op(self):

        def test_cpu_kernel():
            paddle.set_device('cpu')
            tensor = paddle.to_tensor(self.inputs)
            for axis in self.axises:
                value_expect, indice_expect = cal_mode(self.inputs, axis)
                v, inds = paddle.mode(tensor, axis)
                np.testing.assert_allclose(v.numpy(), value_expect, rtol=1e-05)

                value_expect, indice_expect = cal_mode(self.inputs,
                                                       axis,
                                                       keepdim=True)
                v, inds = paddle.mode(tensor, axis, keepdim=True)
                np.testing.assert_allclose(v.numpy(), value_expect, rtol=1e-05)

        def test_gpu_kernel():
            paddle.set_device('gpu')
            tensor = paddle.to_tensor(self.inputs)
            for axis in self.axises:
                value_expect, indice_expect = cal_mode(self.inputs, axis)
                v, inds = paddle.mode(tensor, axis)
                np.testing.assert_allclose(v.numpy(), value_expect, rtol=1e-05)

                value_expect, indice_expect = cal_mode(self.inputs,
                                                       axis,
                                                       keepdim=True)
                v, inds = paddle.mode(tensor, axis, keepdim=True)
                np.testing.assert_allclose(v.numpy(), value_expect, rtol=1e-05)

        paddle.disable_static()
        test_cpu_kernel()
        if fluid.core.is_compiled_with_cuda():
            test_gpu_kernel()


class TestModeOpErrors(unittest.TestCase):

    def setUp(self):
        self.x = paddle.uniform([2, 10, 20, 25], dtype='float32')

        def test_dim_range_error():
            self.x.mode(axis=5)

        self.assertRaises(ValueError, test_dim_range_error)


class TestModeOpInStatic(unittest.TestCase):

    def setUp(self):
        np.random.seed(666)
        self.input_data = np.ceil(np.random.random((2, 10, 10)) * 1000,
                                  dtype=np.float64)

    def test_run_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program(),
                                         paddle.static.Program()):
            input_tensor = paddle.static.data(name="x",
                                              shape=[2, 10, 10],
                                              dtype="float64")

            result = paddle.mode(input_tensor, axis=1)
            expect_value = cal_mode(self.input_data, axis=1)[0]
            exe = paddle.static.Executor(paddle.CPUPlace())
            paddle_result = exe.run(feed={"x": self.input_data},
                                    fetch_list=[result])[0]
            np.testing.assert_allclose(paddle_result, expect_value, rtol=1e-05)


if __name__ == '__main__':
    unittest.main()
