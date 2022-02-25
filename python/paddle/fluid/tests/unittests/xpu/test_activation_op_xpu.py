#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import sys
sys.path.append("..")

import paddle

from op_test import OpTest
from op_test_xpu import XPUOpTest
from xpu.get_test_cover_info import create_test_class, get_xpu_op_support_types, XPUOpTestWrapper

paddle.enable_static()


class TestActivationOPBase(XPUOpTest):
    def setUp(self):
        self.place = paddle.XPUPlace(0)
        self.init_dtype()
        self.set_case()

    def set_case(self):
        self.op_type = 'exp'

        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = np.exp(x)
        self.attrs = {'use_xpu': True}
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output_with_place(self.place)

    def test_check_grad(self):
        self.check_grad_with_place(self.place, ['X'], 'Out')


class XPUTestExpOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'exp'
        self.use_dynamic_create_class = False

    class XPUTestExp(TestActivationOPBase):
        def set_case(self):
            self.op_type = 'exp'
            self.dtype = self.in_type

            x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
            out = np.exp(x)
            self.attrs = {'use_xpu': True}
            self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
            self.outputs = {'Out': out}


support_types = get_xpu_op_support_types('exp')
for stype in support_types:
    create_test_class(globals(), XPUTestExpOP, stype)


class XPUTestSigmoidOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'sigmoid'
        self.use_dynamic_create_class = False

    class XPUTestSigmoid(TestActivationOPBase):
        def set_case(self):
            self.op_type = "sigmoid"
            self.dtype = self.in_type

            x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
            out = 1 / (1 + np.exp(-x))
            self.attrs = {'use_xpu': True}
            self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
            self.outputs = {'Out': out}


support_types = get_xpu_op_support_types('sigmoid')
for stype in support_types:
    create_test_class(globals(), XPUTestSigmoidOP, stype)


class XPUTestTanhOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'tanh'
        self.use_dynamic_create_class = False

    class XPUTestTanh(TestActivationOPBase):
        def set_case(self):
            self.op_type = "tanh"
            self.dtype = self.in_type

            x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
            out = np.tanh(x)
            self.attrs = {'use_xpu': True}
            self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
            self.outputs = {'Out': out}


support_types = get_xpu_op_support_types('tanh')
for stype in support_types:
    create_test_class(globals(), XPUTestTanhOP, stype)


class XPUTestSqrtOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'sqrt'
        self.use_dynamic_create_class = False

    class XPUTestSqrt(TestActivationOPBase):
        def set_case(self):
            self.op_type = "sqrt"
            self.dtype = self.in_type

            x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
            out = np.sqrt(x)

            self.attrs = {'use_xpu': True}
            self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
            self.outputs = {'Out': out}


support_types = get_xpu_op_support_types('sqrt')
for stype in support_types:
    create_test_class(globals(), XPUTestSqrtOP, stype)


class XPUTestAbsOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'abs'
        self.use_dynamic_create_class = False

    class XPUTestAbs(TestActivationOPBase):
        def set_case(self):
            self.op_type = "abs"
            self.dtype = self.in_type

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


support_types = get_xpu_op_support_types('abs')
for stype in support_types:
    create_test_class(globals(), XPUTestAbsOP, stype)


class XPUTestReluOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'relu'
        self.use_dynamic_create_class = False

    class XPUTestRelu(TestActivationOPBase):
        def set_case(self):
            self.op_type = "relu"
            self.dtype = self.in_type

            x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
            # The same reason with TestAbs
            x[np.abs(x) < 0.005] = 0.02
            out = np.maximum(x, 0)

            self.attrs = {'use_xpu': True}
            self.inputs = {'X': x}
            self.outputs = {'Out': out}


support_types = get_xpu_op_support_types('relu')
for stype in support_types:
    create_test_class(globals(), XPUTestReluOP, stype)


class XPUTestGeluOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'gelu'
        self.use_dynamic_create_class = False

    class XPUTestGelu(TestActivationOPBase):
        def set_case(self):
            self.op_type = "gelu"
            self.dtype = self.in_type

            approximate = False
            x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
            out = gelu(x, approximate)

            self.inputs = {'X': x}
            self.outputs = {'Out': out}
            self.attrs = {"approximate": approximate, 'use_xpu': True}


support_types = get_xpu_op_support_types('gelu')
for stype in support_types:
    create_test_class(globals(), XPUTestGeluOP, stype)


def gelu(x, approximate):
    from scipy.special import erf
    if approximate:
        y_ref = 0.5 * x * (1.0 + np.tanh(
            np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    else:
        y_ref = 0.5 * x * (1 + erf(x / np.sqrt(2)))
    return y_ref.astype(x.dtype)


class XPUTestHardSwishOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'hard_swish'
        self.use_dynamic_create_class = False

    class XPUTestHardSwish(TestActivationOPBase):
        def set_case(self):
            self.op_type = "hard_swish"
            self.dtype = self.in_type

            x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
            offset = 3.0
            threshold = 6.0
            scale = 6.0
            out = hard_swish(x, offset, threshold, scale)

            self.inputs = {'X': x}
            self.outputs = {'Out': out}
            self.attrs = {'use_xpu': True}


support_types = get_xpu_op_support_types('hard_swish')
for stype in support_types:
    create_test_class(globals(), XPUTestHardSwishOP, stype)


def hard_swish(x, offset, threshold, scale):
    y_ref = np.minimum(threshold, np.maximum(0, x + offset)) * x / scale
    return y_ref.astype(x.dtype)


class XPUTestLogOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'log'
        self.use_dynamic_create_class = False

    class XPUTestLog(TestActivationOPBase):
        def set_case(self):
            self.op_type = "log"
            self.dtype = self.in_type

            x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
            out = np.log(x)

            self.attrs = {'use_xpu': True}
            self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
            self.outputs = {'Out': out}


support_types = get_xpu_op_support_types('log')
for stype in support_types:
    create_test_class(globals(), XPUTestLogOP, stype)


class XPUTestSquareOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'square'
        self.use_dynamic_create_class = False

    class XPUTestSquare(TestActivationOPBase):
        def set_case(self):
            self.op_type = "square"
            self.dtype = self.in_type

            x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
            out = np.square(x)

            self.attrs = {'use_xpu': True}
            self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
            self.outputs = {'Out': out}


support_types = get_xpu_op_support_types('square')
for stype in support_types:
    create_test_class(globals(), XPUTestSquareOP, stype)


class XPUTestPowOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'pow'
        self.use_dynamic_create_class = False

    class XPUTestPow(TestActivationOPBase):
        def set_case(self):
            self.op_type = "pow"
            self.dtype = self.in_type

            x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
            out = np.power(x, 3)

            self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
            self.attrs = {'factor': 3.0, 'use_xpu': True}
            self.outputs = {'Out': out}


support_types = get_xpu_op_support_types('pow')
for stype in support_types:
    create_test_class(globals(), XPUTestPowOP, stype)


class XPUTestLeakyReluOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'leaky_relu'
        self.use_dynamic_create_class = False

    class XPUTestLeakyRelu(TestActivationOPBase):
        def set_case(self):
            self.op_type = "leaky_relu"
            self.dtype = self.in_type

            x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
            alpha = np.random.uniform(
                0,
                1, )
            out = leaky_relu(x, alpha)

            self.inputs = {'X': x}
            self.outputs = {'Out': out}
            self.attrs = {'use_xpu': True, 'alpha': alpha}


support_types = get_xpu_op_support_types('leaky_relu')
for stype in support_types:
    create_test_class(globals(), XPUTestLeakyReluOP, stype)


def leaky_relu(x, alpha):
    if (alpha < 1):
        y_ref = np.maximum(x, alpha * x)
    else:
        y_ref = np.minimum(x, alpha * x)
    return y_ref.astype(x.dtype)


class XPUTestReciprocalOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'reciprocal'
        self.use_dynamic_create_class = False

    class XPUTestRecipocal(TestActivationOPBase):
        def set_case(self):
            self.op_type = "reciprocal"
            self.dtype = self.in_type

            np.random.seed(1024)
            x = np.random.uniform(1, 2, [1111, 1117]).astype(self.dtype)
            out = np.reciprocal(x)

            self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
            self.outputs = {'Out': out}
            self.attrs = {'use_xpu': True}


support_types = get_xpu_op_support_types('reciprocal')
for stype in support_types:
    create_test_class(globals(), XPUTestReciprocalOP, stype)


class XPUTestSoftPlusOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'softplus'
        self.use_dynamic_create_class = False

    class XPUTestSoftPlusBase(TestActivationOPBase):
        def set_case(self):
            self.op_type = "softplus"
            self.dtype = self.in_type

            self.init_config()
            beta = np.random.uniform(0, 1)
            threshold = np.random.uniform(0, 1)
            out = ref_softplus(self.x, beta, threshold)

            self.inputs = {'X': self.x}
            self.outputs = {'Out': out}
            self.attrs = {'use_xpu': True, 'beta': beta, 'threshold': threshold}

        def init_config(self):
            self.x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)

    class XPUTestSoftPlus2(XPUTestSoftPlusBase):
        def init_config(self):
            self.x = np.random.uniform(-2, 2, [1024, 8]).astype(self.dtype)

    class XPUTestSoftPlus3(XPUTestSoftPlusBase):
        def init_config(self):
            self.x = np.random.uniform(-2, 2,
                                       [4, 512, 15, 15]).astype(self.dtype)

    class XPUTestSoftPlus4(XPUTestSoftPlusBase):
        def init_config(self):
            self.x = np.random.uniform(-2, 2,
                                       [4, 256, 22, 22]).astype(self.dtype)


support_types = get_xpu_op_support_types('softplus')
for stype in support_types:
    create_test_class(globals(), XPUTestSoftPlusOP, stype)


def ref_softplus(x, beta=1, threshold=20):
    x_beta = beta * x
    out = np.select([x_beta <= threshold, x_beta > threshold],
                    [np.log(1 + np.exp(x_beta)) / beta, x])
    return out


if __name__ == "__main__":
    unittest.main()
