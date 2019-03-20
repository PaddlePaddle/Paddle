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
import paddle.fluid.core as core
from op_test import OpTest
from scipy.special import expit, erf


class TestActivation(OpTest):
    def setUp(self):
        self.op_type = "exp"
        self.dtype = np.float32
        self.init_dtype()
        self.init_kernel_type()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.exp(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)

    def init_dtype(self):
        self.dtype = np.float32

    def init_kernel_type(self):
        pass


class TestSigmoid(TestActivation):
    def setUp(self):
        self.op_type = "sigmoid"
        self.init_dtype()

        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = 1 / (1 + np.exp(-x))

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.01)


class TestLogSigmoid(TestActivation):
    def setUp(self):
        self.op_type = "logsigmoid"
        self.init_dtype()

        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = np.log(1 / (1 + np.exp(-x)))

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.008)


class TestTanh(TestActivation):
    def setUp(self):
        self.op_type = "tanh"
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.tanh(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)


class TestAtan(TestActivation):
    def setUp(self):
        self.op_type = "atan"
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.arctan(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)


class TestTanhShrink(TestActivation):
    def setUp(self):
        self.op_type = "tanh_shrink"
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [10, 17]).astype(self.dtype)
        out = x - np.tanh(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.008)


class TestHardShrink(TestActivation):
    def setUp(self):
        self.op_type = "hard_shrink"
        self.init_dtype()

        threshold = 0.5
        x = np.random.uniform(-1, 1, [4, 4]).astype(self.dtype)
        out = np.copy(x)
        out[(out >= -threshold) & (out <= threshold)] = 0

        self.attrs = {'lambda': threshold}
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.005)


class TestSoftShrink(TestActivation):
    def setUp(self):
        self.op_type = "softshrink"
        self.init_dtype()

        lambda_val = 0.1
        x = np.random.uniform(0.25, 10, [4, 4]).astype(self.dtype)
        out = np.copy(x)
        out = (out < -lambda_val) * (out + lambda_val) + (out > lambda_val) * (
            out - lambda_val)

        self.attrs = {'lambda': lambda_val}
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)


class TestSqrt(TestActivation):
    def setUp(self):
        self.op_type = "sqrt"
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.sqrt(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)


class TestAbs(TestActivation):
    def setUp(self):
        self.op_type = "abs"
        self.init_dtype()

        x = np.random.uniform(-1, 1, [4, 4]).astype(self.dtype)
        # Because we set delta = 0.005 in calculating numeric gradient,
        # if x is too small, such as 0.002, x_neg will be -0.003
        # x_pos will be 0.007, so the numeric gradient is inaccurate.
        # we should avoid this
        x[np.abs(x) < 0.005] = 0.02
        out = np.abs(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)


class TestCeil(TestActivation):
    def setUp(self):
        self.op_type = "ceil"
        self.init_dtype()

        x = np.random.uniform(-1, 1, [4, 4]).astype(self.dtype)
        out = np.ceil(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    # The same reason with TestFloor
    def test_check_grad(self):
        pass


class TestFloor(TestActivation):
    def setUp(self):
        self.op_type = "floor"
        self.init_dtype()

        x = np.random.uniform(-1, 1, [4, 4]).astype(self.dtype)
        out = np.floor(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    # the gradient on floor, ceil, round is undefined.
    # we return zero as gradient, but the numpy return nan 
    # The same reason with TestFloor
    def test_check_grad(self):
        pass


class TestCos(TestActivation):
    def setUp(self):
        self.op_type = "cos"
        self.init_dtype()

        x = np.random.uniform(-1, 1, [4, 4]).astype(self.dtype)
        out = np.cos(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)


class TestAcos(TestActivation):
    def setUp(self):
        self.op_type = "acos"
        self.init_dtype()

        x = np.random.uniform(-1, 1, [4, 4]).astype(self.dtype)
        out = np.arccos(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)


class TestSin(TestActivation):
    def setUp(self):
        self.op_type = "sin"
        self.init_dtype()

        x = np.random.uniform(-1, 1, [4, 4]).astype(self.dtype)
        out = np.sin(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)


class TestAsin(TestActivation):
    def setUp(self):
        self.op_type = "asin"
        self.init_dtype()

        x = np.random.uniform(-1, 1, [4, 4]).astype(self.dtype)
        out = np.arcsin(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)


class TestRound(TestActivation):
    def setUp(self):
        self.op_type = "round"
        self.init_dtype()

        x = np.random.uniform(-1, 1, [4, 4]).astype(self.dtype)
        out = np.round(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        pass


class TestRelu(TestActivation):
    def setUp(self):
        self.op_type = "relu"
        self.init_dtype()

        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        # The same reason with TestAbs
        x[np.abs(x) < 0.005] = 0.02
        out = np.maximum(x, 0)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)


class TestGelu(TestActivation):
    def setUp(self):
        self.op_type = "gelu"
        self.init_dtype()

        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)


class TestBRelu(TestActivation):
    def setUp(self):
        self.op_type = "brelu"
        self.init_dtype()

        x = np.random.uniform(-1, 1, [4, 4]).astype(self.dtype)
        t_min = 1.0
        t_max = 4.0
        # The same with TestAbs
        x[np.abs(x - t_min) < 0.005] = t_min + 0.02
        x[np.abs(x - t_max) < 0.005] = t_max + 0.02
        t = np.copy(x)
        t[t < t_min] = t_min
        t[t > t_max] = t_max

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.attrs = {'t_min': t_min, 't_max': t_max}
        self.outputs = {'Out': t}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.02)


class TestRelu6(TestActivation):
    def setUp(self):
        self.op_type = "relu6"
        self.init_dtype()

        x = np.random.uniform(-1, 1, [4, 10]).astype(self.dtype)
        threshold = 6.0
        # The same with TestAbs
        x[np.abs(x) < 0.005] = 0.02
        x[np.abs(x - threshold) < 0.005] = threshold + 0.02
        out = np.minimum(np.maximum(x, 0), threshold)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.attrs = {'threshold': threshold}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.02)


class TestSoftRelu(TestActivation):
    def setUp(self):
        self.op_type = "soft_relu"
        self.init_dtype()

        x = np.random.uniform(-3, 3, [4, 4]).astype(self.dtype)
        threshold = 2.0
        # The same reason with TestAbs
        x[np.abs(x - threshold) < 0.005] = threshold + 0.02
        x[np.abs(x + threshold) < 0.005] = -threshold + 0.02
        t = np.copy(x)
        t[t < -threshold] = -threshold
        t[t > threshold] = threshold
        out = np.log((np.exp(t) + 1))

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.attrs = {'threshold': threshold}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.02)


class TestELU(TestActivation):
    def setUp(self):
        self.op_type = "elu"
        self.init_dtype()

        x = np.random.uniform(-3, 3, [4, 4]).astype(self.dtype)
        alpha = 1.
        out = np.maximum(0, x) + np.minimum(0, alpha * (np.exp(x) - 1))
        # Note: unlike other Relu extensions, point 0 on standard ELU function (i.e. alpha = 1)
        # is differentiable, so we can skip modifications like x[np.abs(x) < 0.005] = 0.02 here
        self.inputs = {'X': x}
        self.attrs = {'alpha': alpha}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.02)


class TestReciprocal(TestActivation):
    def setUp(self):
        self.op_type = "reciprocal"
        self.init_dtype()

        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        out = np.reciprocal(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.01)


class TestLog(TestActivation):
    def setUp(self):
        self.op_type = "log"
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.log(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)


class TestSquare(TestActivation):
    def setUp(self):
        self.op_type = "square"
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.square(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)


class TestPow(TestActivation):
    def setUp(self):
        self.op_type = "pow"
        self.init_dtype()

        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        out = np.power(x, 3)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.attrs = {'factor': 3.0}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.02)


class TestSTanh(TestActivation):
    def setUp(self):
        self.op_type = "stanh"
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        scale_a = 2.0 / 3.0
        scale_b = 1.7159
        out = scale_b * np.tanh(x * scale_a)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.attrs = {'scale_a': scale_a, 'scale_b': scale_b}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)


class TestSoftplus(TestActivation):
    def setUp(self):
        self.op_type = "softplus"
        self.init_dtype()
        self.dtype = np.float64

        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = np.log(1 + np.exp(x))

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)


class TestSoftsign(TestActivation):
    def setUp(self):
        self.op_type = "softsign"
        self.init_dtype()

        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = np.divide(x, 1 + np.abs(x))

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)


class TestThresholdedRelu(TestActivation):
    def setUp(self):
        self.op_type = "thresholded_relu"
        self.init_dtype()

        threshold = 0.25
        self.relative_error = 0.005
        X = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)

        # Same reason as TestAbs
        X[np.abs(X - threshold) < self.relative_error] = threshold + 0.2
        out = (X > threshold) * X

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(X)}
        self.attrs = {'threshold': threshold}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=self.relative_error)


class TestHardSigmoid(TestActivation):
    def setUp(self):
        self.op_type = "hard_sigmoid"
        self.init_dtype()

        self.relative_error = 0.002

        X = np.random.uniform(-5, 5, [2, 2]).astype("float32")
        slope = 0.2
        offset = 0.5
        lower_threshold = -offset / slope
        upper_threshold = (1 - offset) / slope

        # Same reason as TestAbs
        X[np.abs(X - lower_threshold) < self.relative_error] = \
            lower_threshold + 0.2
        X[np.abs(X - upper_threshold) < self.relative_error] = \
            upper_threshold - 0.2

        temp = X * slope + offset
        out = np.maximum(0.0, np.minimum(1.0, temp))

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(X)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.002)


class TestSwish(TestActivation):
    def setUp(self):
        self.op_type = "swish"
        self.init_dtype()

        X = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        beta = 2.3
        out = X * expit(beta * X)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(X)}
        self.attrs = {'beta': beta}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.008)


#------------------ Test Cudnn Activation----------------------
def create_test_act_cudnn_class(parent, atol=1e-3, grad_atol=1e-3):
    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestActCudnn(parent):
        def init_kernel_type(self):
            self.attrs = {"use_cudnn": True}

    cls_name = "{0}_{1}".format(parent.__name__, "cudnn")
    TestActCudnn.__name__ = cls_name
    globals()[cls_name] = TestActCudnn


create_test_act_cudnn_class(TestRelu)
create_test_act_cudnn_class(TestRelu6)
create_test_act_cudnn_class(TestSigmoid)
create_test_act_cudnn_class(TestTanh)


#------------------ Test Fp16 ----------------------
def create_test_act_fp16_class(parent,
                               atol=1e-3,
                               grad_check=True,
                               grad_atol=0.80):
    @unittest.skipIf(not core.is_compiled_with_cuda(),
                     "core is not compiled with CUDA")
    class TestActFp16(parent):
        def init_dtype(self):
            self.dtype = np.float16

        def test_check_output(self):
            place = core.CUDAPlace(0)
            support_fp16 = core.is_float16_supported(place)
            if support_fp16:
                self.check_output_with_place(place, atol=atol)

        def test_check_grad(self):
            place = core.CUDAPlace(0)
            support_fp16 = core.is_float16_supported(place)
            if support_fp16 and grad_check:
                self.check_grad_with_place(
                    place, ['X'], 'Out', max_relative_error=grad_atol)

    cls_name = "{0}_{1}".format(parent.__name__, "fp16")
    TestActFp16.__name__ = cls_name
    globals()[cls_name] = TestActFp16


create_test_act_fp16_class(TestActivation)
create_test_act_fp16_class(TestSigmoid)
create_test_act_fp16_class(TestLogSigmoid)
create_test_act_fp16_class(TestTanh)
create_test_act_fp16_class(TestTanhShrink)
create_test_act_fp16_class(TestHardShrink)
create_test_act_fp16_class(TestSoftShrink)
create_test_act_fp16_class(TestSqrt)
create_test_act_fp16_class(TestAbs)
create_test_act_fp16_class(TestCeil, grad_check=False)
create_test_act_fp16_class(TestFloor, grad_check=False)
create_test_act_fp16_class(TestCos, grad_atol=0.85)
create_test_act_fp16_class(TestAcos, grad_atol=0.85)
create_test_act_fp16_class(TestSin)
create_test_act_fp16_class(TestAsin)
create_test_act_fp16_class(TestAtan)
create_test_act_fp16_class(TestRound, grad_check=False)
create_test_act_fp16_class(TestRelu)
create_test_act_fp16_class(TestGelu)
create_test_act_fp16_class(TestBRelu)
create_test_act_fp16_class(TestRelu6)
create_test_act_fp16_class(TestSoftRelu)
create_test_act_fp16_class(TestELU)
create_test_act_fp16_class(TestReciprocal)
create_test_act_fp16_class(TestLog)
create_test_act_fp16_class(TestSquare)
create_test_act_fp16_class(TestPow, atol=5e-2)
create_test_act_fp16_class(TestSTanh, grad_atol=0.9)
create_test_act_fp16_class(TestSoftplus)
create_test_act_fp16_class(TestSoftsign)
create_test_act_fp16_class(TestThresholdedRelu)
create_test_act_fp16_class(TestHardSigmoid)
create_test_act_fp16_class(TestSwish)

if __name__ == "__main__":
    unittest.main()
