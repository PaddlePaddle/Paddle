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


def p_norm(x, axis, porder, keepdims=False):
    if axis is None: axis = -1
    xp = np.power(np.abs(x), porder)
    s = np.sum(xp, axis=axis, keepdims=keepdims)
    r = np.power(s, 1.0 / porder)
    return r


def frobenius_norm(x, axis=None, keepdims=False):
    if isinstance(axis, list): axis = tuple(axis)
    if axis is None: axis = (-2, -1)
    r = np.linalg.norm(x, ord='fro', axis=axis, keepdims=keepdims)
    return r


class TestFrobeniusNormOp(OpTest):
    def setUp(self):
        self.op_type = "frobenius_norm"
        self.init_test_case()
        x = (np.random.random(self.shape) + 1.0).astype(self.dtype)
        norm = frobenius_norm(x, self.axis, self.keepdim)
        self.reduce_all = (len(self.axis) == len(self.shape))
        self.inputs = {'X': x}
        self.attrs = {
            'dim': list(self.axis),
            'keep_dim': self.keepdim,
            'reduce_all': self.reduce_all
        }
        self.outputs = {'Out': norm}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')

    def init_test_case(self):
        self.shape = [2, 3, 4, 5]
        self.axis = (1, 2)
        self.keepdim = False
        self.dtype = "float64"


class TestFrobeniusNormOp2(TestFrobeniusNormOp):
    def init_test_case(self):
        self.shape = [5, 5, 5]
        self.axis = (0, 1)
        self.keepdim = True
        self.dtype = "float32"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


class TestPnormOp(OpTest):
    def setUp(self):
        self.op_type = "p_norm"
        self.init_test_case()
        x = (np.random.random(self.shape) + 0.5).astype(self.dtype)
        norm = p_norm(x, self.axis, self.porder, self.keepdim)
        self.inputs = {'X': x}
        self.attrs = {
            'epsilon': self.epsilon,
            'axis': self.axis,
            'keepdim': self.keepdim,
            'porder': float(self.porder)
        }
        self.outputs = {'Out': norm}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')

    def init_test_case(self):
        self.shape = [2, 3, 4, 5]
        self.axis = 1
        self.epsilon = 1e-12
        self.porder = 2.0
        self.keepdim = False
        self.dtype = "float64"


class TestPnormOp2(TestPnormOp):
    def init_test_case(self):
        self.shape = [3, 20, 3]
        self.axis = 2
        self.epsilon = 1e-12
        self.porder = 2.0
        self.keepdim = True
        self.dtype = "float32"

    def test_check_grad(self):
        self.check_grad(['X'], 'Out')


def run_out(self, p, axis, shape_x, shape_y, dtype):
    with fluid.program_guard(fluid.Program()):
        data1 = fluid.data(name="X", shape=shape_x, dtype=dtype)
        data2 = fluid.data(name="Y", shape=shape_y, dtype=dtype)
        out = paddle.norm(input=data1, p=p, axis=axis, out=data2)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        result = exe.run(feed={"X": np.random.rand(*shape_x).astype(dtype)},
                         fetch_list=[data2, out])
        self.assertEqual((result[0] == result[1]).all(), True)


def run_fro(self, p, axis, shape_x, dtype):
    with fluid.program_guard(fluid.Program()):
        data = fluid.data(name="X", shape=shape_x, dtype=dtype)
        out = paddle.norm(input=data, p=p, axis=axis)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        np_input = (np.random.rand(*shape_x) + 1.0).astype(dtype)
        expected_result = frobenius_norm(np_input, axis=axis)
        result, = exe.run(feed={"X": np_input}, fetch_list=[out])
    self.assertEqual((np.abs(result - expected_result) < 1e-6).all(), True)


def run_pnorm(self, p, axis, shape_x, dtype):
    with fluid.program_guard(fluid.Program()):
        data = fluid.data(name="X", shape=shape_x, dtype=dtype)
        out = paddle.norm(input=data, p=p, axis=axis)
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        np_input = (np.random.rand(*shape_x) + 1.0).astype(dtype)
        expected_result = p_norm(np_input, porder=p, axis=axis).astype(dtype)
        result, = exe.run(feed={"X": np_input}, fetch_list=[out])
    self.assertEqual((np.abs(result - expected_result) < 1e-6).all(), True)


class API_NormTest(unittest.TestCase):
    def test_output_result(self):
        run_out(self, p=2, axis=1, shape_x=[3, 4], shape_y=[3], dtype="float32")
        run_out(
            self,
            p='fro',
            axis=None,
            shape_x=[3, 4],
            shape_y=[1],
            dtype="float32")

    def test_basic(self):
        run_fro(self, p='fro', axis=None, shape_x=[3, 3, 4], dtype="float32")
        run_fro(self, p='fro', axis=[0, 1], shape_x=[3, 3, 4], dtype="float64")
        run_pnorm(self, p=2, axis=None, shape_x=[3, 4], dtype="float32")
        run_pnorm(self, p=2, axis=1, shape_x=[3, 4], dtype="float64")

    def test_name(self):
        with fluid.program_guard(fluid.Program()):
            x = fluid.data(name="x", shape=[10, 10], dtype="float32")
            y_1 = paddle.norm(x, p='fro', name='frobenius_name')
            y_2 = paddle.norm(x, p=2, name='pnorm_name')
            self.assertEqual(('frobenius_name' in y_1.name), True)
            self.assertEqual(('pnorm_name' in y_2.name), True)

    def test_errors(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):

            def err_dtype(p, shape_x, xdtype, out=None):
                data = fluid.data(shape=shape_x, dtype=xdtype)
                paddle.norm(data, p=p, out=out)

            self.assertRaises(TypeError, err_dtype, "fro", [2, 2], "int64")
            out = fluid.data(name="out", shape=[1], dtype="int64")
            self.assertRaises(TypeError, err_dtype, "fro", [2, 2], "float64",
                              out)
            self.assertRaises(TypeError, err_dtype, 2, [10], "int64")
            self.assertRaises(TypeError, err_dtype, 2, [10], "float64", out)

            data = fluid.data(name="data_2d", shape=[2, 2], dtype="float64")
            self.assertRaises(ValueError, paddle.norm, data, p="unsupport norm")
            self.assertRaises(ValueError, paddle.norm, data, p=[1])
            self.assertRaises(ValueError, paddle.norm, data, p=[1], axis=-1)
            self.assertRaises(
                ValueError, paddle.norm, data, p='unspport', axis=[-2, -1])
            data = fluid.data(name="data_3d", shape=[2, 2, 2], dtype="float64")
            self.assertRaises(
                ValueError, paddle.norm, data, p='unspport', axis=[-2, -1])
            self.assertRaises(
                ValueError, paddle.norm, data, p='unspport', axis=[-3, -2, -1])


if __name__ == '__main__':
    unittest.main()
