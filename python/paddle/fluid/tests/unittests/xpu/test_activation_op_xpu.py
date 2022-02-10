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
import unittest
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest
from op_test_xpu import XPUOpTest
from scipy.special import expit, erf
import paddle
import paddle.fluid as fluid
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.fluid import compiler, Program, program_guard


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPUActivation(XPUOpTest):
    def setUp(self):
        self.op_type = "exp"
        self.init_dtype()
        self.init_kernel_type()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.exp(x)

        self.attrs = {'use_xpu': True}
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_output_with_place(place, atol=1e-3)

    def init_kernel_type(self):
        pass


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPUSigmoid(TestXPUActivation):
    def setUp(self):
        self.op_type = "sigmoid"
        self.init_dtype()

        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = 1 / (1 + np.exp(-x))

        self.attrs = {'use_xpu': True}
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['X'], 'Out')


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPUTanh(TestXPUActivation):
    def setUp(self):
        self.op_type = "tanh"
        self.init_dtype()
        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.tanh(x)

        self.attrs = {'use_xpu': True}
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['X'], 'Out')


class TestXPUTanhFP16(TestXPUActivation):
    def setUp(self):
        self.op_type = "tanh"
        self.init_dtype()
        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.tanh(x)

        self.attrs = {'use_xpu': True}
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_grad(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['X'], 'Out')


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPUSqrt(TestXPUActivation):
    def setUp(self):
        self.op_type = "sqrt"
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.sqrt(x)

        self.attrs = {'use_xpu': True}
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['X'], 'Out')


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPUAbs(TestXPUActivation):
    def setUp(self):
        self.op_type = "abs"
        self.init_dtype()

        x = np.random.uniform(-1, 1, [4, 25]).astype(self.dtype)
        # Because we set delta = 0.005 in calculating numeric gradient,
        # if x is too small, such as 0.002, x_neg will be -0.003
        # x_pos will be 0.007, so the numeric gradient is inaccurate.
        # we should avoid this
        x[np.abs(x) < 0.005] = 0.02
        out = np.abs(x)

        self.attrs = {'use_xpu': True}
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['X'], 'Out')


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPURelu(TestXPUActivation):
    def setUp(self):
        self.op_type = "relu"
        self.init_dtype()

        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        # The same reason with TestAbs
        x[np.abs(x) < 0.005] = 0.02
        out = np.maximum(x, 0)

        self.attrs = {'use_xpu': True}
        self.inputs = {'X': x}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['X'], 'Out')


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPUGelu(TestXPUActivation):
    def setUp(self):
        self.op_type = "gelu"
        self.init_dtype()
        approximate = False
        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = gelu(x, approximate)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {"approximate": approximate, 'use_xpu': True}

    def test_check_grad(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['X'], 'Out')


class TestXPUGelu(TestXPUActivation):
    def setUp(self):
        self.op_type = "gelu"
        self.init_dtype()
        approximate = False
        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = gelu(x, approximate)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {"approximate": approximate, 'use_xpu': True}

    def init_dtype(self):
        self.dtype = np.float16

    def test_check_grad(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['X'], 'Out')


def gelu(x, approximate):
    if approximate:
        y_ref = 0.5 * x * (1.0 + np.tanh(
            np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    else:
        y_ref = 0.5 * x * (1 + erf(x / np.sqrt(2)))
    return y_ref.astype(x.dtype)


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPUHardSwish(TestXPUActivation):
    def setUp(self):
        self.op_type = "hard_swish"
        self.init_dtype()
        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        offset = 3.0
        threshold = 6.0
        scale = 6.0
        out = hard_swish(x, offset, threshold, scale)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {'use_xpu': True}

    def test_check_grad(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['X'], 'Out')


def hard_swish(x, offset, threshold, scale):
    y_ref = np.minimum(threshold, np.maximum(0, x + offset)) * x / scale
    return y_ref.astype(x.dtype)


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPULog(TestXPUActivation):
    def setUp(self):
        self.op_type = "log"
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.log(x)

        self.attrs = {'use_xpu': True}
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPUSquare(TestXPUActivation):
    def setUp(self):
        self.op_type = "square"
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.square(x)

        self.attrs = {'use_xpu': True}
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['X'], 'Out')


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPUPow(TestXPUActivation):
    def setUp(self):
        self.op_type = "pow"
        self.init_dtype()

        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        out = np.power(x, 3)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.attrs = {'factor': 3.0, 'use_xpu': True}
        self.outputs = {'Out': out}


@unittest.skipIf(not paddle.is_compiled_with_xpu(),
                 "core is not compiled with XPU")
class TestXPULeakyRelu(TestXPUActivation):
    def setUp(self):
        self.op_type = "leaky_relu"
        self.init_dtype()
        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        alpha = np.random.uniform(
            0,
            1, )
        out = leaky_relu(x, alpha)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {'use_xpu': True, 'alpha': alpha}

    def test_check_grad(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['X'], 'Out')


def leaky_relu(x, alpha):
    if (alpha < 1):
        y_ref = np.maximum(x, alpha * x)
    else:
        y_ref = np.minimum(x, alpha * x)
    return y_ref.astype(x.dtype)


class TestXPUReciprocal(TestXPUActivation):
    def setUp(self):
        self.op_type = "reciprocal"
        self.init_dtype()

        np.random.seed(1024)
        x = np.random.uniform(1, 2, [1111, 1117]).astype(self.dtype)
        out = np.reciprocal(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {'use_xpu': True}

    def test_check_grad(self):
        if paddle.is_compiled_with_xpu():
            place = paddle.XPUPlace(0)
            self.check_grad_with_place(place, ['X'], 'Out')


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
