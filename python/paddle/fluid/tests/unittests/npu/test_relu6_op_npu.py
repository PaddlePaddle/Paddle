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
from scipy.special import expit, erf

from paddle.fluid.tests.unittests.op_test import OpTest, convert_float_to_uint16, skip_check_grad_ci
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle.fluid import compiler, Program, program_guard

paddle.enable_static()

def ref_relu6(x, threshold=6.0):
    out = np.copy(x)
    out[np.abs(x - threshold) < 0.005] = threshold + 0.02
    out = np.minimum(np.maximum(x, 0), threshold)
    return out

class TestActivation(OpTest):
    def setUp(self):
        self.op_type = "exp"
        self.init_dtype()
        self.init_kernel_type()

        np.random.seed(2049)
        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.exp(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')

    def init_dtype(self):
        self.dtype = np.float64

    def init_kernel_type(self):
        pass


class TestRelu6(TestActivation):
    def setUp(self):
        self.op_type = "relu6"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(-1, 10, [10, 12]).astype(self.dtype)
        x[np.abs(x) < 0.005] = 0.02
        out = ref_relu6(x)

        self.inputs = {'X': x}
        self.attrs = {'threshold': 6.0}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestRelu6API(unittest.TestCase):
    # test paddle.nn.ReLU6, paddle.nn.functional.relu6
    def setUp(self):
        np.random.seed(1024)
        self.x_np = np.random.uniform(-1, 10, [10, 12]).astype(np.float64)
        self.x_np[np.abs(self.x_np) < 0.005] = 0.02
        self.place=paddle.NPUPlace(0) 

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.fluid.data('X', self.x_np.shape, self.x_np.dtype)
            out1 = F.relu6(x)
            relu6 = paddle.nn.ReLU6()
            out2 = relu6(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = ref_relu6(self.x_np)
        for r in res:
            self.assertEqual(np.allclose(out_ref, r), True)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.relu6(x)
        relu6 = paddle.nn.ReLU6()
        out2 = relu6(x)
        out_ref = ref_relu6(self.x_np)
        for r in [out1, out2]:
            self.assertEqual(np.allclose(out_ref, r.numpy()), True)
        paddle.enable_static()

    def test_fluid_api(self):
        paddle.enable_static()
        with fluid.program_guard(fluid.Program()):
            x = fluid.data('X', self.x_np.shape, self.x_np.dtype)
            out = fluid.layers.relu6(x)
            exe = fluid.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out])
        out_ref = ref_relu6(self.x_np)
        self.assertEqual(np.allclose(out_ref, res[0]), True)

    def test_errors(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, F.relu6, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.fluid.data(
                name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, F.relu6, x_int32)
            # support the input dtype is float16
            x_fp16 = paddle.fluid.data(
                name='x_fp16', shape=[12, 10], dtype='float16')
            F.relu6(x_fp16)
