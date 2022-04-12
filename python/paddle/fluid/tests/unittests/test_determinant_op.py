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

import unittest
import numpy as np
from op_test import OpTest
import paddle
import paddle.nn.functional as F
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.tensor as tensor
from paddle.fluid.framework import _test_eager_guard

paddle.enable_static()


class TestDeterminantOp(OpTest):
    def setUp(self):
        self.python_api = paddle.linalg.det
        self.init_data()
        self.op_type = "determinant"
        self.outputs = {'Out': self.target}

    def test_check_output(self):
        self.check_output(check_eager=True)

    def test_check_grad(self):
        self.check_grad(['Input'], ['Out'], check_eager=True)

    def init_data(self):
        np.random.seed(0)
        self.case = np.random.rand(3, 3, 3, 5, 5).astype('float64')
        self.inputs = {'Input': self.case}
        self.target = np.linalg.det(self.case)


class TestDeterminantOpCase1(TestDeterminantOp):
    def init_data(self):
        np.random.seed(0)
        self.case = np.random.rand(10, 10).astype('float32')
        self.inputs = {'Input': self.case}
        self.target = np.linalg.det(self.case)


class TestDeterminantOpCase2(TestDeterminantOp):
    def init_data(self):
        np.random.seed(0)
        # not invertible matrix
        self.case = np.ones([4, 2, 4, 4]).astype('float64')
        self.inputs = {'Input': self.case}
        self.target = np.linalg.det(self.case)


class TestDeterminantAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.shape = [3, 3, 5, 5]
        self.x = np.random.random(self.shape).astype(np.float32)
        self.place = paddle.CPUPlace()

    def test_api_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', self.shape)
            out = paddle.linalg.det(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x}, fetch_list=[out])
        out_ref = np.linalg.det(self.x)

        for out in res:
            self.assertEqual(np.allclose(out, out_ref, rtol=1e-03), True)

    def test_api_dygraph(self):
        paddle.disable_static(self.place)
        x_tensor = paddle.to_tensor(self.x)
        out = paddle.linalg.det(x_tensor)
        out_ref = np.linalg.det(self.x)
        self.assertEqual(np.allclose(out.numpy(), out_ref, rtol=1e-03), True)
        paddle.enable_static()

    def test_eager(self):
        with _test_eager_guard():
            self.test_api_dygraph()


class TestSlogDeterminantOp(OpTest):
    def setUp(self):
        self.op_type = "slogdeterminant"
        self.init_data()
        self.outputs = {'Out': self.target}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        # the slog det's grad value is always huge
        self.check_grad(['Input'], ['Out'], max_relative_error=0.1)

    def init_data(self):
        np.random.seed(0)
        self.case = np.random.rand(4, 5, 5).astype('float64')
        self.inputs = {'Input': self.case}
        self.target = np.array(np.linalg.slogdet(self.case))


class TestSlogDeterminantOpCase1(TestSlogDeterminantOp):
    def init_data(self):
        np.random.seed(0)
        self.case = np.random.rand(2, 2, 5, 5).astype(np.float32)
        self.inputs = {'Input': self.case}
        self.target = np.array(np.linalg.slogdet(self.case))


class TestSlogDeterminantAPI(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.shape = [3, 3, 5, 5]
        self.x = np.random.random(self.shape).astype(np.float32)
        self.place = paddle.CPUPlace()

    def test_api_static(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', self.shape)
            out = paddle.linalg.slogdet(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x}, fetch_list=[out])
        out_ref = np.array(np.linalg.slogdet(self.x))
        for out in res:
            self.assertEqual(np.allclose(out, out_ref, rtol=1e-03), True)

    def test_api_dygraph(self):
        paddle.disable_static(self.place)
        x_tensor = paddle.to_tensor(self.x)
        out = paddle.linalg.slogdet(x_tensor)
        out_ref = np.array(np.linalg.slogdet(self.x))
        self.assertEqual(np.allclose(out.numpy(), out_ref, rtol=1e-03), True)
        paddle.enable_static()


if __name__ == '__main__':
    unittest.main()
