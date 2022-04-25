# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import numpy as np
import unittest
import sys
sys.path.append("..")
from op_test import OpTest
import paddle
import paddle.fluid as fluid
from paddle.fluid import core
import paddle.nn.functional as F

from test_log_softmax import ref_log_softmax, ref_log_softmax_grad

paddle.enable_static()


class TestLogSoftmaxNPUOp(OpTest):
    def setUp(self):
        self.set_npu()
        self.place = paddle.NPUPlace(0)
        self.op_type = "log_softmax"
        self.dtype = np.float32
        self.shape = [2, 3, 4, 5]
        self.axis = -1
        self.set_attrs()
        self.set_dtype()
        x = np.random.uniform(0.1, 1., self.shape).astype(self.dtype)
        out = np.apply_along_axis(ref_log_softmax, self.axis, x)
        self.x_grad = ref_log_softmax_grad(x, self.axis)
        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {'axis': self.axis}

    def set_npu(self):
        self.__class__.use_npu = True
        self.__class__.no_need_check_grad = True

    def set_attrs(self):
        pass

    def set_dtype(self):
        pass

    def test_check_output(self):
        if self.dtype == np.float16:
            self.check_output_with_place(self.place, atol=1e-2)
        else:
            self.check_output_with_place(self.place)

    def test_check_grad(self):
        if self.dtype == np.float16:
            self.check_grad_with_place(
                self.place, ['X'], ['Out'],
                user_defined_grads=[self.x_grad],
                max_relative_error=0.02)
        else:
            self.check_grad_with_place(
                self.place, ['X'], ['Out'], user_defined_grads=[self.x_grad])


def test_class(op_type, typename):
    class TestLogSoftmaxShape(TestLogSoftmaxNPUOp):
        def set_attrs(self):
            self.shape = [12, 10]

        def set_dtype(self):
            self.dtype = typename

    cls_name = "{0}_{1}_1".format(op_type, typename)
    TestLogSoftmaxShape.__name__ = cls_name
    globals()[cls_name] = TestLogSoftmaxShape


def test_class2(op_type, typename):
    class TestLogSoftmaxAxis(TestLogSoftmaxNPUOp):
        def set_attrs(self):
            self.axis = 0

        def set_dtype(self):
            self.dtype = typename

    cls_name = "{0}_{1}_2".format(op_type, typename)

    TestLogSoftmaxAxis.__name__ = cls_name
    globals()[cls_name] = TestLogSoftmaxAxis


for _typename in {np.float32, np.float16}:
    test_class("logsoftmax", _typename)
    test_class2("logsoftmax", _typename)


class TestNNLogSoftmaxAPI(unittest.TestCase):
    def setUp(self):
        self.x_shape = [2, 3, 4, 5]
        self.x = np.random.uniform(-1., 1., self.x_shape).astype(np.float32)
        self.place = paddle.NPUPlace(0) \
            if paddle.fluid.core.is_compiled_with_npu() \
            else paddle.CPUPlace()

    def check_api(self, axis=-1):
        ref_out = np.apply_along_axis(ref_log_softmax, axis, self.x)

        logsoftmax = paddle.nn.LogSoftmax(axis)
        # test static api
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data(name='x', shape=self.x_shape)
            y = logsoftmax(x)
            exe = paddle.static.Executor(self.place)
            out = exe.run(feed={'x': self.x}, fetch_list=[y])
        self.assertTrue(np.allclose(out[0], ref_out))

        # test dygrapg api
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        y = logsoftmax(x)
        self.assertTrue(np.allclose(y.numpy(), ref_out))
        paddle.enable_static()

    def test_check_api(self):
        for axis in [-1, 1]:
            self.check_api(axis)


class TestNNFunctionalLogSoftmaxAPI(unittest.TestCase):
    def setUp(self):
        self.x_shape = [2, 3, 4, 5]
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(np.float32)
        self.place = paddle.NPUPlace(0) \
            if paddle.fluid.core.is_compiled_with_npu() \
            else paddle.CPUPlace()

    def check_api(self, axis=-1, dtype=None):
        x = self.x.copy()
        if dtype is not None:
            x = x.astype(dtype)
        ref_out = np.apply_along_axis(ref_log_softmax, axis, x)
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data(name='x', shape=self.x_shape)
            y = F.log_softmax(x, axis, dtype)
            exe = paddle.static.Executor(self.place)
            out = exe.run(feed={'x': self.x}, fetch_list=[y])
        self.assertTrue(np.allclose(out[0], ref_out))

        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x)
        y = F.log_softmax(x, axis, dtype)
        self.assertTrue(np.allclose(y.numpy(), ref_out), True)
        paddle.enable_static()

    def test_check_api(self):
        for axis in [-1, 1]:
            self.check_api(axis)


if __name__ == '__main__':
    unittest.main()
