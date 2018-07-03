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
import paddle.fluid.core as core
from op_test import OpTest
from scipy.special import expit


class TestExp(OpTest):
    def setUp(self):
        self.op_type = "exp"
        self.dtype = np.float32
        self.init_dtype()

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
        pass


class TestFP16Exp(TestExp):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestSigmoid(OpTest):
    def setUp(self):
        self.op_type = "sigmoid"
        self.dtype = np.float32
        self.init_dtype()

        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = 1 / (1 + np.exp(-x))

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.01)

    def init_dtype(self):
        pass


class TestFP16Sigmoid(TestSigmoid):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestLogSigmoid(OpTest):
    def setUp(self):
        self.op_type = "logsigmoid"
        self.dtype = np.float32
        self.init_dtype()

        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = np.log(1 / (1 + np.exp(-x)))

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.008)

    def init_dtype(self):
        pass


class TestFP16LogSigmoid(TestLogSigmoid):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestTanh(OpTest):
    def setUp(self):
        self.op_type = "tanh"
        self.dtype = np.float32
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.tanh(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)

    def init_dtype(self):
        pass


class TestFP16Tanh(TestTanh):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestTanhShrink(OpTest):
    def setUp(self):
        self.op_type = "tanh_shrink"
        self.dtype = np.float32
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [10, 17]).astype(self.dtype)
        out = x - np.tanh(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.008)

    def init_dtype(self):
        pass


class TestFP16TanhShrink(TestTanhShrink):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestHardShrink(OpTest):
    def setUp(self):
        self.op_type = "hard_shrink"
        self.dtype = np.float32
        self.init_dtype()

        threshold = 0.5
        x = np.random.uniform(-1, 1, [4, 4]).astype(self.dtype)
        out = np.copy(x)
        out[(out >= -threshold) & (out <= threshold)] = 0

        self.attrs = {'lambda': threshold}
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.005)

    def init_dtype(self):
        pass


class TestFP16HardShrink(TestHardShrink):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestSoftShrink(OpTest):
    def setUp(self):
        self.op_type = "softshrink"
        self.dtype = np.float32
        self.init_dtype()

        lambda_val = 0.1
        x = np.random.uniform(0.25, 10, [4, 4]).astype(self.dtype)
        out = np.copy(x)
        out = (out < -lambda_val) * (out + lambda_val) + (out > lambda_val) * (
            out - lambda_val)

        self.attrs = {'lambda': lambda_val}
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)

    def init_dtype(self):
        pass


class TestFP16SoftShrink(TestSoftShrink):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestSqrt(OpTest):
    def setUp(self):
        self.op_type = "sqrt"
        self.dtype = np.float32
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.sqrt(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)

    def init_dtype(self):
        pass


class TestFP16Sqrt(TestSqrt):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestAbs(OpTest):
    def setUp(self):
        self.op_type = "abs"
        self.dtype = np.float32
        self.init_dtype()

        x = np.random.uniform(-1, 1, [4, 4]).astype(self.dtype)
        # Because we set delta = 0.005 in caculating numeric gradient,
        # if x is too small, such as 0.002, x_neg will be -0.003
        # x_pos will be 0.007, so the numeric gradient is unaccurate.
        # we should avoid this
        x[np.abs(x) < 0.005] = 0.02
        out = np.abs(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)

    def init_dtype(self):
        pass


class TestFP16Abs(TestAbs):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestCeil(OpTest):
    def setUp(self):
        self.op_type = "ceil"
        self.dtype = np.float32
        self.init_dtype()

        x = np.random.uniform(-1, 1, [4, 4]).astype(self.dtype)
        out = np.ceil(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    # The same reason with TestFloor

    def init_dtype(self):
        pass


class TestFP16Ceil(TestCeil):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestFloor(OpTest):
    def setUp(self):
        self.op_type = "floor"
        self.dtype = np.float32
        self.init_dtype()

        x = np.random.uniform(-1, 1, [4, 4]).astype(self.dtype)
        out = np.floor(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    # the gradient on floor, ceil, round is undefined.
    # we return zero as gradient, but the numpy return nan 

    def init_dtype(self):
        pass


class TestFP16Floor(TestFloor):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestCos(OpTest):
    def setUp(self):
        self.op_type = "cos"
        self.dtype = np.float32
        self.init_dtype()

        x = np.random.uniform(-1, 1, [4, 4]).astype(self.dtype)
        out = np.cos(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)

    def init_dtype(self):
        pass


class TestFP16Cos(TestCos):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestSin(OpTest):
    def setUp(self):
        self.op_type = "sin"
        self.dtype = np.float32
        self.init_dtype()

        x = np.random.uniform(-1, 1, [4, 4]).astype(self.dtype)
        out = np.sin(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)

    def init_dtype(self):
        pass


class TestFP16Sin(TestSin):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestRound(OpTest):
    def setUp(self):
        self.op_type = "round"
        self.dtype = np.float32
        self.init_dtype()

        x = np.random.uniform(-1, 1, [4, 4]).astype(self.dtype)
        out = np.round(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def init_dtype(self):
        pass


class TestFP16Round(TestRound):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestRelu(OpTest):
    def setUp(self):
        self.op_type = "relu"
        self.dtype = np.float32
        self.init_dtype()

        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        # The same reason with TestAbs
        x[np.abs(x) < 0.005] = 0.02
        out = np.maximum(x, 0)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)

    def init_dtype(self):
        pass


class TestFP16Relu(TestRelu):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestBRelu(OpTest):
    def setUp(self):
        self.op_type = "brelu"
        self.dtype = np.float32
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

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.02)

    def init_dtype(self):
        pass


class TestFP16BRelu(TestBRelu):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestRelu6(OpTest):
    def setUp(self):
        self.op_type = "relu6"
        self.dtype = np.float32
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

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.02)

    def init_dtype(self):
        pass


class TestFP16Relu6(TestRelu6):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestSoftRelu(OpTest):
    def setUp(self):
        self.op_type = "soft_relu"
        self.dtype = np.float32
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

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.02)

    def init_dtype(self):
        pass


class TestFP16SoftRelu(TestSoftRelu):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestELU(OpTest):
    def setUp(self):
        self.op_type = "elu"
        self.dtype = np.float32
        self.init_dtype()

        x = np.random.uniform(-3, 3, [4, 4]).astype(self.dtype)
        alpha = 1.
        out = np.maximum(0, x) + np.minimum(0, alpha * (np.exp(x) - 1))
        # Note: unlike other Relu extensions, point 0 on standard ELU function (i.e. alpha = 1)
        # is differentiable, so we can skip modifications like x[np.abs(x) < 0.005] = 0.02 here
        self.inputs = {'X': x}
        self.attrs = {'alpha': alpha}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.02)

    def init_dtype(self):
        pass


class TestFP16ELU(TestELU):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestReciprocal(OpTest):
    def setUp(self):
        self.op_type = "reciprocal"
        self.dtype = np.float32
        self.init_dtype()

        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        out = np.reciprocal(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.01)

    def init_dtype(self):
        pass


class TestFP16Reciprocal(TestReciprocal):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestLog(OpTest):
    def setUp(self):
        self.op_type = "log"
        self.dtype = np.float32
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.log(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)

    def init_dtype(self):
        pass


class TestFP16Log(TestLog):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestSquare(OpTest):
    def setUp(self):
        self.op_type = "square"
        self.dtype = np.float32
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.square(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)

    def init_dtype(self):
        pass


class TestFP16Square(TestSquare):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestPow(OpTest):
    def setUp(self):
        self.op_type = "pow"
        self.dtype = np.float32
        self.init_dtype()

        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        out = np.power(x, 3)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.attrs = {'factor': 3.0}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.02)

    def init_dtype(self):
        pass


class TestFP16Pow(TestPow):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=5e-2)


class TestSTanh(OpTest):
    def setUp(self):
        self.op_type = "stanh"
        self.dtype = np.float32
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        scale_a = 2.0 / 3.0
        scale_b = 1.7159
        out = scale_b * np.tanh(x * scale_a)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.attrs = {'scale_a': scale_a, 'scale_b': scale_b}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)

    def init_dtype(self):
        pass


class TestFP16STanh(TestSTanh):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestSoftplus(OpTest):
    def setUp(self):
        self.op_type = "softplus"
        self.dtype = np.float64
        self.init_dtype()

        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = np.log(1 + np.exp(x))

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)

    def init_dtype(self):
        pass


class TestFP16Softplus(TestSoftplus):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestSoftsign(OpTest):
    def setUp(self):
        self.op_type = "softsign"
        self.dtype = np.float32
        self.init_dtype()

        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = np.divide(x, 1 + np.abs(x))

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.007)

    def init_dtype(self):
        pass


class TestFP16Softsign(TestSoftsign):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestThresholdedRelu(OpTest):
    def setUp(self):
        self.op_type = "thresholded_relu"
        self.dtype = np.float32
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

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=self.relative_error)

    def init_dtype(self):
        pass


class TestFP16ThresholdedRelu(TestThresholdedRelu):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestHardSigmoid(OpTest):
    def setUp(self):
        self.op_type = "hard_sigmoid"
        self.dtype = np.float32
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

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.002)

    def init_dtype(self):
        pass


class TestFP16HardSigmoid(TestHardSigmoid):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestSwish(OpTest):
    def setUp(self):
        self.op_type = "swish"
        self.dtype = np.float32
        self.init_dtype()

        X = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        beta = 2.3
        out = X * expit(beta * X)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(X)}
        self.attrs = {'beta': beta}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.008)

    def init_dtype(self):
        pass


class TestFP16Swish(TestSwish):
    def init_dtype(self):
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
