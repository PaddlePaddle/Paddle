# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid.dygraph as dg


class TestKronOp(OpTest):
    def setUp(self):
        self.op_type = "kron"
        self.dtype = self._init_dtype()
        x = np.random.uniform(size=(10, 10)).astype(self.dtype)
        y = np.random.uniform(size=(10, 10)).astype(self.dtype)
        out_ref = np.kron(x, y)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out_ref}

    def _init_dtype(self):
        return "float64"

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X', 'Y'], 'Out')

    def test_check_grad_ignore_x(self):
        self.check_grad(['Y'], 'Out', no_grad_set=set('X'))

    def test_check_grad_ignore_y(self):
        self.check_grad(['X'], 'Out', no_grad_set=set('Y'))


class TestKronOp2(TestKronOp):
    def setUp(self):
        self.op_type = "kron"
        self.dtype = self._init_dtype()
        x = np.random.uniform(size=(5, 5, 4)).astype(self.dtype)
        y = np.random.uniform(size=(10, 10)).astype(self.dtype)
        out_ref = np.kron(x, y)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out_ref}


class TestKronOp3(TestKronOp):
    def setUp(self):
        self.op_type = "kron"
        self.dtype = self._init_dtype()
        x = np.random.uniform(size=(10, 10)).astype(self.dtype)
        y = np.random.uniform(size=(5, 5, 4)).astype(self.dtype)
        out_ref = np.kron(x, y)
        self.inputs = {'X': x, 'Y': y}
        self.outputs = {'Out': out_ref}


class TestKronLayer(unittest.TestCase):
    def test_case(self):
        a = np.random.randn(10, 10).astype(np.float64)
        b = np.random.randn(10, 10).astype(np.float64)

        place = fluid.CPUPlace()
        with dg.guard(place):
            a_var = dg.to_variable(a)
            b_var = dg.to_variable(b)
            c_var = paddle.kron(a_var, b_var)
            np.testing.assert_allclose(c_var.numpy(), np.kron(a, b))

    def test_case_with_output(self):
        a = np.random.randn(10, 10).astype(np.float64)
        b = np.random.randn(10, 10).astype(np.float64)

        main = fluid.Program()
        start = fluid.Program()
        with fluid.unique_name.guard():
            with fluid.program_guard(main, start):
                a_var = fluid.data("a", [-1, -1], dtype="float64")
                b_var = fluid.data("b", [-1, -1], dtype="float64")
                out_var = fluid.layers.create_tensor("float64", "c")
                paddle.kron(a_var, b_var, out=out_var)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(start)
        c, = exe.run(main, feed={'a': a, 'b': b}, fetch_list=[out_var])
        np.testing.assert_allclose(c, np.kron(a, b))


if __name__ == '__main__':
    unittest.main()
