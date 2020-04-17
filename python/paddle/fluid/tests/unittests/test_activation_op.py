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
import paddle
import paddle.fluid as fluid
import paddle.nn as nn
import paddle.nn.functional as functional
from paddle.fluid import compiler, Program, program_guard


class TestSqrtOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of sqrt op must be Variable or numpy.ndarray.
            in1 = 1
            self.assertRaises(TypeError, fluid.layers.sqrt, in1)
            # The input dtype of sqrt op must be float16, float32, float64.
            in2 = fluid.layers.data(
                name='input2', shape=[12, 10], dtype="int32")
            self.assertRaises(TypeError, fluid.layers.sqrt, in2)

            in3 = fluid.layers.data(
                name='input3', shape=[12, 10], dtype="float16")
            fluid.layers.sqrt(x=in3)


class TestActivation(OpTest):
    def setUp(self):
        self.op_type = "exp"
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
        self.check_grad(['X'], 'Out')

    def init_dtype(self):
        self.dtype = np.float64

    def init_kernel_type(self):
        pass


class TestParameter(object):
    def test_out(self):
        with fluid.program_guard(fluid.Program()):
            data = fluid.layers.data(name="X", shape=[1])
            out = eval("paddle.%s(data, out=data)" % self.op_type)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            result = exe.run(feed={"X": np.array([0.1])},
                             fetch_list=[data, out])
            self.assertEqual(result[0], result[1])

    def test_out_name(self):
        with fluid.program_guard(fluid.Program()):
            data = fluid.layers.data(name="X", shape=[1])
            out = eval("paddle.%s(data, name='Y', out=data)" % self.op_type)
            place = fluid.CPUPlace()
            exe = fluid.Executor(place)
            result = exe.run(feed={"X": np.array([0.1])},
                             fetch_list=[data, out])
            self.assertEqual(result[0], result[1])

    def test_dygraph(self):
        with fluid.dygraph.guard():
            np_x = np.array([0.1])
            x = fluid.dygraph.to_variable(np_x)
            z = eval("paddle.%s(x).numpy()" % self.op_type)
            z_expected = eval("np.%s(np_x)" % self.op_type)
            self.assertEqual(z, z_expected)


class TestSigmoid(TestActivation):
    def setUp(self):
        self.op_type = "sigmoid"
        self.init_dtype()

        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = 1 / (1 + np.exp(-x))

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def init_dtype(self):
        self.dtype = np.float32

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


class TestTanh(TestActivation, TestParameter):
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
        self.check_grad(['X'], 'Out')

    def init_dtype(self):
        #TODO If dtype is float64, the output (Out) has diff at CPUPlace
        # when using and not using inplace. Therefore, set dtype as float32
        # for now.
        self.dtype = np.float32


class TestAtan(TestActivation, TestParameter):
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
        self.check_grad(['X'], 'Out')

    def test_dygraph(self):
        with fluid.dygraph.guard():
            np_x = np.array([0.1])
            x = fluid.dygraph.to_variable(np_x)
            z = paddle.atan(x).numpy()
            z_expected = np.arctan(np_x)
            self.assertEqual(z, z_expected)


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
        self.check_grad(['X'], 'Out')


class TestHardShrink(TestActivation):
    def setUp(self):
        self.op_type = "hard_shrink"
        self.init_dtype()

        threshold = 0.5
        x = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype) * 10
        out = np.copy(x)
        out[(out >= -threshold) & (out <= threshold)] = 0

        self.attrs = {'lambda': threshold}
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestHardShrinkOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, fluid.layers.hard_shrink, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = fluid.data(name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, fluid.layers.hard_shrink, x_int32)
            # support the input dtype is float16
            x_fp16 = fluid.data(name='x_fp16', shape=[12, 10], dtype='float16')
            fluid.layers.hard_shrink(x_fp16)


class TestSoftShrink(TestActivation):
    def setUp(self):
        self.op_type = "softshrink"
        self.init_dtype()

        lambda_val = 0.1
        x = np.random.uniform(0.25, 10, [10, 12]).astype(self.dtype)
        out = np.copy(x)
        out = (out < -lambda_val) * (out + lambda_val) + (out > lambda_val) * (
            out - lambda_val)

        self.attrs = {'lambda': lambda_val}
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestSoftShrinkOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, fluid.layers.softshrink, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = fluid.data(name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, fluid.layers.softshrink, x_int32)
            # support the input dtype is float16
            x_fp16 = fluid.data(name='x_fp16', shape=[12, 10], dtype='float16')
            fluid.layers.softshrink(x_fp16)


class TestSqrt(TestActivation, TestParameter):
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
        self.check_grad(['X'], 'Out')


class TestRsqrt(TestActivation):
    def setUp(self):
        self.op_type = "rsqrt"
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [10, 12]).astype(self.dtype) * 10
        out = 1.0 / np.sqrt(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', max_relative_error=0.0005)


class TestAbs(TestActivation):
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

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestCeil(TestActivation):
    def setUp(self):
        self.op_type = "ceil"
        self.init_dtype()

        x = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype)
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

        x = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype)
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

        x = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype)
        out = np.cos(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestAcos(TestActivation):
    def setUp(self):
        self.op_type = "acos"
        self.init_dtype()

        x = np.random.uniform(-0.95, 0.95, [10, 12]).astype(self.dtype)
        out = np.arccos(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestSin(TestActivation, TestParameter):
    def setUp(self):
        self.op_type = "sin"
        self.init_dtype()

        x = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype)
        out = np.sin(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestAsin(TestActivation):
    def setUp(self):
        self.op_type = "asin"
        self.init_dtype()

        x = np.random.uniform(-0.95, 0.95, [10, 12]).astype(self.dtype)
        out = np.arcsin(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestRound(TestActivation):
    def setUp(self):
        self.op_type = "round"
        self.init_dtype()

        x = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype)
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
        self.check_grad(['X'], 'Out')


class TestReluOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, fluid.layers.relu, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = fluid.data(name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, fluid.layers.relu, x_int32)
            # support the input dtype is float16
            x_fp16 = fluid.layers.data(
                name='x_fp16', shape=[12, 10], dtype='float16')
            fluid.layers.relu(x_fp16)


class TestLeakyRelu(TestActivation):
    def setUp(self):
        self.op_type = "leaky_relu"
        self.init_dtype()

        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        # The same reason with TestAbs
        x[np.abs(x) < 0.005] = 0.02
        out = np.maximum(x, 0.02 * x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestLeakyReluOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, fluid.layers.leaky_relu, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = fluid.data(name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, fluid.layers.leaky_relu, x_int32)
            # support the input dtype is float32
            x_fp16 = fluid.layers.data(
                name='x_fp16', shape=[12, 10], dtype='float32')
            fluid.layers.leaky_relu(x_fp16)


def gelu(x, approximate):
    if approximate:
        y_ref = 0.5 * x * (1.0 + np.tanh(
            np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    else:
        y_ref = 0.5 * x * (1 + erf(x / np.sqrt(2)))
    return y_ref.astype(x.dtype)


class TestGeluApproximate(TestActivation):
    def setUp(self):
        self.op_type = "gelu"
        self.init_dtype()
        approximate = True
        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = gelu(x, approximate)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {"approximate": approximate}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestGelu(TestActivation):
    def setUp(self):
        self.op_type = "gelu"
        self.init_dtype()
        approximate = False
        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = gelu(x, approximate)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {"approximate": approximate}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestBRelu(TestActivation):
    def setUp(self):
        self.op_type = "brelu"
        self.init_dtype()

        x = np.random.uniform(-5, 10, [10, 12]).astype(self.dtype)
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
        self.check_grad(['X'], 'Out')


class TestBReluOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, fluid.layers.brelu, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = fluid.data(name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, fluid.layers.brelu, x_int32)
            # support the input dtype is float16
            x_fp16 = fluid.layers.data(
                name='x_fp16', shape=[12, 10], dtype='float16')
            fluid.layers.brelu(x_fp16)


class TestRelu6(TestActivation):
    def setUp(self):
        self.op_type = "relu6"
        self.init_dtype()

        x = np.random.uniform(-1, 10, [10, 12]).astype(self.dtype)
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
        self.check_grad(['X'], 'Out')


class TestRelu6OpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, fluid.layers.relu6, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = fluid.data(name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, fluid.layers.relu6, x_int32)
            # support the input dtype is float16
            x_fp16 = fluid.data(name='x_fp16', shape=[12, 10], dtype='float16')
            fluid.layers.relu6(x_fp16)


class TestHardSwish(TestActivation):
    def setUp(self):
        self.op_type = 'hard_swish'
        self.init_dtype()

        x = np.random.uniform(-6, 6, [10, 12]).astype(self.dtype)
        threshold = 6.0
        scale = 6.0
        offset = 3.0
        #the same with TestAbs
        x[np.abs(x + offset) < 0.005] = 0.02
        x[np.abs(x - threshold + offset) < 0.005] = threshold - offset + 0.02
        out = x * np.minimum(np.maximum(x + offset, 0), threshold) / scale

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.attrs = {'threshold': threshold, 'scale': scale, 'offset': offset}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestHardSwishOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, fluid.layers.hard_swish, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = fluid.data(name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, fluid.layers.hard_swish, x_int32)
            # support the input dtype is float16
            x_fp16 = fluid.data(name='x_fp16', shape=[12, 10], dtype='float16')
            fluid.layers.hard_swish(x_fp16)


class TestSoftRelu(TestActivation):
    def setUp(self):
        self.op_type = "soft_relu"
        self.init_dtype()

        x = np.random.uniform(-3, 3, [4, 4]).astype(self.dtype)
        threshold = 2.0
        # The same reason with TestAbs
        x[np.abs(x - threshold) < 0.005] = threshold + 0.02
        x[np.abs(x + threshold) < 0.005] = -threshold - 0.02
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


class TestSoftReluOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, fluid.layers.soft_relu, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = fluid.data(name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, fluid.layers.soft_relu, x_int32)
            # support the input dtype is float16
            x_fp16 = fluid.data(name='x_fp16', shape=[12, 10], dtype='float16')
            fluid.layers.soft_relu(x_fp16)


class TestELU(TestActivation):
    def setUp(self):
        self.op_type = "elu"
        self.init_dtype()

        x = np.random.uniform(-3, 3, [10, 12]).astype(self.dtype)
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
        self.check_grad(['X'], 'Out')


class TestELUOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program(), Program()):
            # The input type of elu_op must be Variable.
            x1 = fluid.create_lod_tensor(
                np.array([[-1]]), [[1]], fluid.CPUPlace())
            self.assertRaises(TypeError, fluid.layers.elu, x1)
            # The input dtype of elu_op must be float16 float32 or float64.
            x2 = fluid.layers.data(name='x2', shape=[4], dtype="int32")
            self.assertRaises(TypeError, fluid.layers.elu, x2)


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
        self.check_grad(['X'], 'Out')

    def test_error(self):
        in1 = fluid.layers.data(
            name="in1", shape=[11, 17], append_batch_size=False, dtype="int32")
        in2 = fluid.layers.data(
            name="in2", shape=[11, 17], append_batch_size=False, dtype="int64")

        self.assertRaises(TypeError, fluid.layers.log, in1)
        self.assertRaises(TypeError, fluid.layers.log, in2)


class TestLog1p(TestActivation):
    def setUp(self):
        self.op_type = "log1p"
        self.init_dtype()

        x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
        out = np.log1p(x)

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(x)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')

    def test_api(self):
        with fluid.program_guard(fluid.Program(), fluid.Program()):
            input_x = np.random.uniform(0.1, 1, [11, 17]).astype("float64")
            data_x = fluid.layers.data(
                name="data_x",
                shape=[11, 17],
                append_batch_size=False,
                dtype="float64")
            res_log1p = fluid.layers.data(
                name="res_log1p",
                shape=[11, 17],
                append_batch_size=False,
                dtype="float64")

            out1 = paddle.log1p(data_x)
            out2 = paddle.log1p(data_x, out=res_log1p)
            exe = fluid.Executor(place=fluid.CPUPlace())
            exe.run(fluid.default_startup_program())
            res1, res_in = exe.run(fluid.default_main_program(),
                                   feed={"data_x": input_x},
                                   fetch_list=[out1, res_log1p])
        expected_res = np.log1p(input_x)
        np.testing.assert_allclose(res1, expected_res)
        np.testing.assert_allclose(res_in, expected_res)

        # dygraph
        with fluid.dygraph.guard():
            np_x = np.random.uniform(0.1, 1, [11, 17]).astype("float64")
            data_x = fluid.dygraph.to_variable(np_x)
            z = paddle.log1p(data_x)
            np_z = z.numpy()
            z_expected = np.array(np.log1p(np_x))
        np.testing.assert_allclose(np_z, z_expected)


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
        self.check_grad(['X'], 'Out')


class TestPow_factor_tensor(TestActivation):
    def setUp(self):
        self.op_type = "pow"
        self.init_dtype()

        x = np.random.uniform(1, 2, [11, 17]).astype(self.dtype)
        out = np.power(x, 3)

        self.inputs = {
            'X': OpTest.np_dtype_to_fluid_dtype(x),
            'FactorTensor': np.array([3.0]).astype("float32")
        }

        self.attrs = {}
        self.outputs = {'Out': out}

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')

    def test_api(self):
        input = np.random.uniform(1, 2, [11, 17]).astype("float32")
        x = fluid.layers.data(
            name="x", shape=[11, 17], append_batch_size=False, dtype="float32")
        res = fluid.layers.data(
            name="res",
            shape=[11, 17],
            append_batch_size=False,
            dtype="float32")

        factor_1 = 2.0
        factor_2 = fluid.layers.fill_constant([1], "float32", 3.0)
        out_1 = fluid.layers.pow(x, factor=factor_1)
        out_2 = fluid.layers.pow(x, factor=factor_2)
        out_3 = paddle.pow(x, factor_1, out=res)
        out_4 = paddle.pow(x, factor_1, name='pow_res')
        out_5 = paddle.pow(x, factor_1, out=res, name='pow_res')
        out_6 = paddle.pow(x, factor_2)
        self.assertEqual(('pow_res' in out_4.name), True)

        exe = fluid.Executor(place=fluid.CPUPlace())
        res_1, res_2, res_3, res, res_6 = exe.run(
            fluid.default_main_program(),
            feed={"x": input},
            fetch_list=[out_1, out_2, out_3, res, out_6])

        assert np.array_equal(res_1, np.power(input, 2))
        assert np.array_equal(res_2, np.power(input, 3))
        assert np.array_equal(res_3, res)
        assert np.array_equal(res_6, np.power(input, 3))

    def test_error(self):
        in1 = fluid.layers.data(
            name="in1", shape=[11, 17], append_batch_size=False, dtype="int32")
        in2 = fluid.layers.data(
            name="in2", shape=[11, 17], append_batch_size=False, dtype="int64")
        in3 = fluid.layers.data(
            name="in3",
            shape=[11, 17],
            append_batch_size=False,
            dtype="float32")
        in4 = fluid.layers.data(
            name="in4",
            shape=[11, 17],
            append_batch_size=False,
            dtype="float64")

        factor_1 = fluid.layers.fill_constant([1], "float64", 3.0)

        self.assertRaises(TypeError, fluid.layers.pow, x=in1, factor=factor_1)
        self.assertRaises(TypeError, fluid.layers.pow, x=in2, factor=factor_1)
        self.assertRaises(TypeError, fluid.layers.pow, x=in3, factor=factor_1)
        self.assertRaises(TypeError, fluid.layers.pow, x=in4, factor=factor_1)


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
        self.check_grad(['X'], 'Out')


class TestSTanhOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, fluid.layers.stanh, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = fluid.data(name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, fluid.layers.stanh, x_int32)
            # support the input dtype is float16
            x_fp16 = fluid.data(name='x_fp16', shape=[12, 10], dtype='float16')
            fluid.layers.stanh(x_fp16)


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
        self.check_grad(['X'], 'Out')


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
        self.check_grad(['X'], 'Out')


class TestThresholdedRelu(TestActivation):
    def setUp(self):
        self.op_type = "thresholded_relu"
        self.init_dtype()

        threshold = 0.25
        self.delta = 0.005
        X = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)

        # Same reason as TestAbs
        X[np.abs(X - threshold) < self.delta] = threshold + 0.2
        out = (X > threshold) * X

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(X)}
        self.attrs = {'threshold': threshold}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestThresholdedReluOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, fluid.layers.thresholded_relu, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = fluid.data(name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, fluid.layers.thresholded_relu, x_int32)
            # support the input dtype is float16
            x_fp16 = fluid.data(name='x_fp16', shape=[12, 10], dtype='float16')
            fluid.layers.thresholded_relu(x_fp16)


class TestHardSigmoid(TestActivation):
    def setUp(self):
        self.op_type = "hard_sigmoid"
        self.init_dtype()

        X = np.random.uniform(-5, 5, [10, 12]).astype("float32")
        slope = 0.2
        offset = 0.5
        lower_threshold = -offset / slope
        upper_threshold = (1 - offset) / slope

        self.delta = 0.005

        # Same reason as TestAbs
        X[(X - lower_threshold) < self.delta] = lower_threshold - 0.02
        X[(X - upper_threshold) < self.delta] = upper_threshold + 0.02

        temp = X * slope + offset
        out = np.maximum(0.0, np.minimum(1.0, temp))

        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(X)}
        self.outputs = {'Out': out}

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out')


class TestHardSigmoidOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, fluid.layers.hard_sigmoid, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = fluid.data(name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, fluid.layers.hard_sigmoid, x_int32)
            # support the input dtype is float16
            x_fp16 = fluid.data(name='x_fp16', shape=[12, 10], dtype='float16')
            fluid.layers.hard_sigmoid(x_fp16)


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


class TestSwishOpError(unittest.TestCase):
    def test_errors(self):
        with program_guard(Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, fluid.layers.swish, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = fluid.data(name='x_int32', shape=[12, 10], dtype='int32')
            self.assertRaises(TypeError, fluid.layers.swish, x_int32)
            # support the input dtype is float16
            x_fp16 = fluid.data(name='x_fp16', shape=[12, 10], dtype='float16')
            fluid.layers.swish(x_fp16)


#------------------ Test Error Activation----------------------
def create_test_error_class(op_type):
    class TestOpErrors(unittest.TestCase):
        def test_errors(self):
            with program_guard(Program(), Program()):
                op = getattr(fluid.layers, op_type)
                # The input dtype of op_type must be float32, float64.
                in1 = fluid.layers.data(
                    name='input2', shape=[12, 10], dtype="int32")
                in2 = fluid.layers.data(
                    name='input3', shape=[12, 10], dtype="int64")
                self.assertRaises(TypeError, op, in1)
                self.assertRaises(TypeError, op, in2)

    cls_name = "{0}_{1}".format(op_type, "test_errors")
    TestOpErrors.__name__ = cls_name
    globals()[cls_name] = TestOpErrors


create_test_error_class('acos')
create_test_error_class('asin')
create_test_error_class('atan')
create_test_error_class('ceil')
create_test_error_class('cos')
create_test_error_class('floor')
create_test_error_class('reciprocal')
create_test_error_class('round')
create_test_error_class('rsqrt')
create_test_error_class('sin')
create_test_error_class('sqrt')
create_test_error_class('tanh')


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
create_test_act_fp16_class(TestLog1p, grad_atol=0.9)
create_test_act_fp16_class(TestSquare)
create_test_act_fp16_class(TestPow, atol=5e-2)
create_test_act_fp16_class(TestPow_factor_tensor, atol=5e-2)
create_test_act_fp16_class(TestSTanh, grad_atol=0.9)
create_test_act_fp16_class(TestSoftplus)
create_test_act_fp16_class(TestSoftsign)
create_test_act_fp16_class(TestThresholdedRelu)
create_test_act_fp16_class(TestHardSigmoid)
create_test_act_fp16_class(TestSwish)
create_test_act_fp16_class(TestHardSwish)


class TestNNReluAPI(unittest.TestCase):
    def setUp(self):
        self.init_data()

    def init_data(self):
        self.x_shape = [10, 12]
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(np.float32)
        self.y = self.ref_forward(self.x)

    def ref_forward(self, x):
        return np.maximum(x, 0)

    def ref_backward(self, y, dy):
        y_t = y.copy()
        y_t[y_t > 0] = 1
        return y_t * dy

    def check_api(self, place=fluid.CPUPlace(), inplace=False):
        main_program = Program()
        myrelu = nn.ReLU(inplace)
        with fluid.program_guard(main_program):
            x = fluid.data(name='x', shape=self.x_shape)
            x.stop_gradient = False
            y = myrelu(x)
            fluid.backward.append_backward(fluid.layers.mean(y))
        exe = fluid.Executor(place)
        out = exe.run(main_program,
                      feed={'x': self.x},
                      fetch_list=[y, y.grad_name, x.grad_name])
        self.assertTrue(np.allclose(out[0], self.y))
        self.assertTrue(np.allclose(out[2], self.ref_backward(self.y, out[1])))

        with fluid.dygraph.guard(place):
            x = fluid.dygraph.to_variable(self.x)
            y = myrelu(x)
        self.assertTrue(np.allclose(y.numpy(), self.y))

    def test_check_api(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for place in places:
            for inplace in [True, False]:
                self.check_api(place, inplace)


class TestNNFunctionalReluAPI(unittest.TestCase):
    def setUp(self):
        self.init_data()

    def init_data(self):
        self.x_shape = [10, 12]
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(np.float32)
        self.y = self.ref_forward(self.x)

    def ref_forward(self, x):
        return np.maximum(x, 0)

    def test_check_api(self):
        main_program = Program()
        with fluid.program_guard(main_program):
            x = fluid.data(name='x', shape=self.x_shape)
            y = functional.relu(x)
        exe = fluid.Executor(fluid.CPUPlace())
        out = exe.run(main_program, feed={'x': self.x}, fetch_list=[y])
        self.assertTrue(np.allclose(out[0], self.y))


class TestNNSigmoidAPI(unittest.TestCase):
    def setUp(self):
        self.init_data()

    def init_data(self):
        self.x_shape = [10, 15]
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(np.float32)
        self.y = self.ref_forward(self.x)

    def ref_forward(self, x):
        return 1 / (1 + np.exp(-x))

    def ref_backward(self, y, dy):
        return dy * y * (1 - y)

    def check_api(self, place=fluid.CPUPlace(), inplace=False):
        main_program = Program()
        mysigmoid = nn.Sigmoid(inplace)
        with fluid.program_guard(main_program):
            x = fluid.data(name='x', shape=self.x_shape)
            x.stop_gradient = False
            y = mysigmoid(x)
            fluid.backward.append_backward(fluid.layers.mean(y))
        exe = fluid.Executor(place)
        out = exe.run(main_program,
                      feed={'x': self.x},
                      fetch_list=[y, y.grad_name, x.grad_name])
        self.assertTrue(np.allclose(out[0], self.y))
        self.assertTrue(np.allclose(out[2], self.ref_backward(self.y, out[1])))

        with fluid.dygraph.guard(place):
            x = fluid.dygraph.to_variable(self.x)
            y = mysigmoid(x)
        self.assertTrue(np.allclose(y.numpy(), self.y))

    def test_check_api(self):
        places = [fluid.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(fluid.CUDAPlace(0))
        for place in places:
            for inplace in [True, False]:
                self.check_api(place, inplace)


class TestNNFunctionalSigmoidAPI(unittest.TestCase):
    def setUp(self):
        self.init_data()

    def init_data(self):
        self.x_shape = [10, 15]
        self.x = np.random.uniform(-1, 1, self.x_shape).astype(np.float32)
        self.y = self.ref_forward(self.x)

    def ref_forward(self, x):
        return 1 / (1 + np.exp(-x))

    def test_check_api(self):
        main_program = Program()
        with fluid.program_guard(main_program):
            x = fluid.data(name='x', shape=self.x_shape)
            y = functional.sigmoid(x)
        exe = fluid.Executor(fluid.CPUPlace())
        out = exe.run(main_program, feed={'x': self.x}, fetch_list=[y])
        self.assertTrue(np.allclose(out[0], self.y))


if __name__ == "__main__":
    unittest.main()
