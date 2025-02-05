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


import os
import unittest
from contextlib import contextmanager

import numpy as np
from get_test_cover_info import (
    XPUOpTestWrapper,
    check_run_big_shape_test,
    create_test_class,
    get_xpu_op_support_types,
)
from op_test import OpTest, convert_float_to_uint16, convert_uint16_to_float
from op_test_xpu import XPUOpTest

import paddle
import paddle.nn.functional as F

paddle.enable_static()


@contextmanager
def dynamic_guard():
    paddle.disable_static()
    try:
        yield
    finally:
        paddle.enable_static()


class TestActivationOPBase(XPUOpTest):
    def setUp(self):
        self.place = paddle.XPUPlace(0)
        self.init_dtype()
        self.set_shape()
        self.set_case()

    def set_shape(self):
        self.shape = [11, 17]

    def set_case(self):
        self.op_type = 'exp'
        x = np.random.uniform(-1, 1, self.shape).astype(self.dtype)
        out = np.exp(x)
        self.attrs = {'use_xpu': True}
        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
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

            x = np.random.uniform(-1, 1, [11, 17])
            if self.dtype == np.uint16:
                new_x = convert_float_to_uint16(x)
            else:
                new_x = x.astype(self.dtype)
            out = np.exp(x)
            self.attrs = {'use_xpu': True}
            self.inputs = {'X': new_x}
            self.outputs = {'Out': out}

    class XPUTestExp_ZeroDIm(TestActivationOPBase):
        def set_shape(self):
            self.shape = []


support_types = get_xpu_op_support_types('exp')
for stype in support_types:
    create_test_class(globals(), XPUTestExpOP, stype)


class XPUTestRoundOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'round'
        self.use_dynamic_create_class = False

    class XPUTestRound(TestActivationOPBase):
        def set_case(self):
            self.op_type = 'round'

            self.init_dtype()
            self.set_shape()
            self.set_decimals()

            np.random.seed(1024)
            x = np.random.uniform(-1, 1, self.shape).astype(self.dtype) * 100
            out = np.round(x, decimals=self.decimals)

            self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
            self.outputs = {'Out': out}
            self.attrs = {'decimals': self.decimals}
            self.convert_input_output()

        def set_shape(self):
            self.shape = [10, 12]

        def set_decimals(self):
            self.decimals = 0

        def test_check_grad(self):
            pass

        def convert_input_output(self):
            if self.dtype == np.uint16:
                self.inputs = {'X': convert_float_to_uint16(self.inputs['X'])}
                self.outputs = {
                    'Out': convert_float_to_uint16(self.outputs['Out'])
                }

    class XPUTestRound_ZeroDIm(XPUTestRound):
        def set_shape(self):
            self.shape = []

    class XPUTestRound_decimals1(XPUTestRound):
        def init_decimals(self):
            self.decimals = 2

        def test_round_api(self):
            with dynamic_guard():
                x_np = (
                    np.random.uniform(-1, 1, self.shape).astype(self.dtype)
                    * 100
                )
                out_expect = np.round(x_np, decimals=self.decimals)
                x_paddle = paddle.to_tensor(
                    x_np, dtype=self.dtype, place=self.place
                )
                y = paddle.round(x_paddle, decimals=self.decimals)
                np.testing.assert_allclose(y.numpy(), out_expect, rtol=1e-3)

    class TestRound_decimals2(XPUTestRound_decimals1):
        def init_decimals(self):
            self.decimals = -1


support_types = get_xpu_op_support_types('round')
for stype in support_types:
    create_test_class(globals(), XPUTestRoundOP, stype)


class XPUTestSiluOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'silu'
        self.use_dynamic_create_class = False

    class XPUTestSilu(TestActivationOPBase):
        def set_case(self):
            self.op_type = "silu"
            self.dtype = self.in_type
            self.init_shape()

            np.random.seed(1024)
            x = np.random.uniform(-1, 1, self.shape)

            if self.dtype == np.uint16:
                # bfloat16 actually
                new_x = convert_float_to_uint16(x)
            else:
                new_x = x.astype(self.dtype)
            out = x / (np.exp(-x) + 1)

            self.inputs = {'X': new_x}
            self.outputs = {'Out': out}
            self.attrs = {'use_xpu': True}

        def test_check_output(self):
            self.set_env()
            self.check_output_with_place(self.place)
            self.delete_env()

        def test_check_grad(self):
            self.set_env()
            self.check_grad_with_place(self.place, ['X'], 'Out')
            self.delete_env()

        def init_shape(self):
            self.shape = [11, 17]

        def set_env(self):
            pass

        def delete_env(self):
            pass

    class TestSilu_ZeroDim(XPUTestSilu):
        def init_shape(self):
            self.shape = []

    class TestSilu_LUT(XPUTestSilu):
        def set_env(self):
            # set "XPU_PADDLE_ACT_LUT" env to enable lut
            os.environ['XPU_PADDLE_ACT_LUT'] = "1"

        def delete_env(self):
            if os.getenv('XPU_PADDLE_ACT_LUT'):
                del os.environ['XPU_PADDLE_ACT_LUT']

    @check_run_big_shape_test()
    class TestSiluLargeShape1(XPUTestSilu):
        def init_shape(self):
            self.shape = [8192, 1728]


class TestSiluAPI(unittest.TestCase):
    # test paddle.nn.Silu, paddle.nn.functional.silu
    def setUp(self):
        self.x_np = np.random.uniform(-1, 1, [11, 17]).astype('float32')
        self.place = paddle.XPUPlace(0)

    def test_static_api(self):
        paddle.enable_static()
        with paddle.static.program_guard(paddle.static.Program()):
            x = paddle.static.data('X', [11, 17])
            out1 = F.silu(x)
            m = paddle.nn.Silu()
            out2 = m(x)
            exe = paddle.static.Executor(self.place)
            res = exe.run(feed={'X': self.x_np}, fetch_list=[out1, out2])
        out_ref = self.x_np / (1 + np.exp(-self.x_np))
        for r in res:
            np.testing.assert_allclose(out_ref, r, rtol=1e-05)

    def test_dygraph_api(self):
        paddle.disable_static(self.place)
        x = paddle.to_tensor(self.x_np)
        out1 = F.silu(x)
        m = paddle.nn.Silu()
        out2 = m(x)
        out_ref = self.x_np / (1 + np.exp(-self.x_np))
        for r in [out1, out2]:
            np.testing.assert_allclose(out_ref, r.numpy(), rtol=1e-05)
        paddle.enable_static()

    def test_errors(self):
        with paddle.static.program_guard(paddle.static.Program()):
            # The input type must be Variable.
            self.assertRaises(TypeError, F.silu, 1)
            # The input dtype must be float16, float32, float64.
            x_int32 = paddle.static.data(
                name='x_int32', shape=[11, 17], dtype='int32'
            )
            self.assertRaises(TypeError, F.silu, x_int32)
            # support the input dtype is float16
            x_fp16 = paddle.static.data(
                name='x_fp16', shape=[11, 17], dtype='float16'
            )
            F.silu(x_fp16)


support_types = get_xpu_op_support_types('silu')
for stype in support_types:
    create_test_class(globals(), XPUTestSiluOP, stype)


class XPUTestSigmoidOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'sigmoid'
        self.use_dynamic_create_class = False

    class XPUTestSigmoid(TestActivationOPBase):
        def set_case(self):
            self.op_type = "sigmoid"
            self.dtype = self.in_type
            self.init_config()
            out = 1 / (1 + np.exp(-self.x))

            self.attrs = {'use_xpu': True}
            self.inputs = {'X': OpTest.np_dtype_to_base_dtype(self.x)}
            self.outputs = {'Out': out}

        def init_config(self):
            self.x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)

    class XPUTestSigmoid_ZeroDIm(XPUTestSigmoid):
        def init_config(self):
            self.x = np.random.uniform(-2, 2, []).astype(self.dtype)

    class XPUTestSigmoid2(XPUTestSigmoid):
        def init_config(self):
            self.x = np.random.uniform(-2, 2, [100]).astype(self.dtype)

    class XPUTestSigmoid3(XPUTestSigmoid):
        def init_config(self):
            self.x = np.random.uniform(-2, 2, [10, 12, 15]).astype(self.dtype)

    class XPUTestSigmoid4(XPUTestSigmoid):
        def init_config(self):
            self.x = np.random.uniform(-2, 2, [19, 19]).astype(self.dtype)

    class XPUTestSigmoid5(XPUTestSigmoid):
        def init_config(self):
            self.x = np.random.uniform(-2, 2, [10, 20, 30, 40]).astype(
                self.dtype
            )


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

            x = np.random.uniform(-1, 1, [11, 17])
            if self.dtype == np.uint16:
                # bfloat16 actually
                new_x = convert_float_to_uint16(x)
            else:
                new_x = x.astype(self.dtype)

            out = np.tanh(x)
            self.attrs = {'use_xpu': True}
            self.inputs = {'X': new_x}
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
            self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
            self.outputs = {'Out': out}


support_types = get_xpu_op_support_types('sqrt')
for stype in support_types:
    create_test_class(globals(), XPUTestSqrtOP, stype)


class XPUTestFloorOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'floor'
        self.use_dynamic_create_class = False

    class XPUTestSqrt(TestActivationOPBase):
        def set_case(self):
            self.op_type = "floor"
            self.dtype = self.in_type

            x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
            out = np.floor(x)

            self.attrs = {'use_xpu': True}
            self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
            self.outputs = {'Out': out}

        def test_check_grad(self):
            self.check_output_with_place(self.place)


support_types = get_xpu_op_support_types('floor')
for stype in support_types:
    create_test_class(globals(), XPUTestFloorOP, stype)


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
            self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
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

            tmp_x = np.random.uniform(-1, 1, [11, 17])
            # The same reason with TestAbs
            tmp_x[np.abs(tmp_x) < 0.005] = 0.02

            if self.dtype == np.uint16:
                # bfloat16 actually
                tmp_out = np.maximum(tmp_x, 0)
                x = convert_float_to_uint16(tmp_x)
                out = convert_float_to_uint16(tmp_out)
            else:
                x = tmp_x.astype(self.dtype)
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

    class XPUTestGeluBase(TestActivationOPBase):
        def set_case(self):
            self.op_type = "gelu"
            self.dtype = self.in_type

            self.init_config()
            out = gelu(self.x, self.approximate)

            self.inputs = {'X': self.x}
            self.outputs = {'Out': out}
            self.attrs = {"approximate": self.approximate, 'use_xpu': True}

        def init_config(self):
            self.approximate = False
            self.x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)

    class XPUTestGelu_ZeroDim(XPUTestGeluBase):
        def init_config(self):
            self.approximate = False
            self.x = np.random.uniform(-2, 2, []).astype(self.dtype)

    class XPUTestGelu1(XPUTestGeluBase):
        def init_config(self):
            self.approximate = True
            self.x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)

    class XPUTestGelu2(XPUTestGeluBase):
        def init_config(self):
            self.approximate = False
            self.x = np.random.uniform(-2, 2, [1024, 8]).astype(self.dtype)

    class XPUTestGelu3(XPUTestGeluBase):
        def init_config(self):
            self.approximate = True
            self.x = np.random.uniform(-2, 2, [4, 512, 15, 15]).astype(
                self.dtype
            )

    class XPUTestGelu4(XPUTestGeluBase):
        def init_config(self):
            self.approximate = False
            self.x = np.random.uniform(-2, 2, [4, 256, 22, 22]).astype(
                self.dtype
            )


support_types = get_xpu_op_support_types('gelu')
for stype in support_types:
    create_test_class(globals(), XPUTestGeluOP, stype)


def gelu(x, approximate):
    from scipy.special import erf

    if approximate:
        y_ref = (
            0.5
            * x
            * (
                1.0
                + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3)))
            )
        )
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
            x = np.random.uniform(0.1, 1, self.shape)
            if self.dtype == np.int16:
                new_x = convert_float_to_uint16(x)
            else:
                new_x = x.astype(self.dtype)
            out = np.log(x)

            self.attrs = {'use_xpu': True}
            self.inputs = {'X': new_x}
            self.outputs = {'Out': out}

    class TestLogCase_ZeroDim(XPUTestLog):
        def set_shape(self):
            self.shape = []

    class TestLogCase1(XPUTestLog):
        def set_shape(self):
            self.shape = [1, 11, 17]

    class TestLogCase2(XPUTestLog):
        def set_shape(self):
            self.shape = [2, 2, 2]

    class TestLogCase3(XPUTestLog):
        def set_shape(self):
            self.shape = [2]

    class TestLogCase4(XPUTestLog):
        def set_shape(self):
            self.shape = [1, 2, 3, 4]


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
            self.init_config()
            if self.dtype == np.uint16:
                # bfloat16 actually
                self.x = convert_float_to_uint16(self.tmp_x)
            else:
                self.x = self.tmp_x.astype(self.dtype)
            out = np.square(self.x)

            self.attrs = {'use_xpu': True}
            self.inputs = {'X': OpTest.np_dtype_to_base_dtype(self.x)}
            self.outputs = {'Out': out}

        def init_config(self):
            self.tmp_x = np.random.uniform(-1, 1, [11, 17])

    class XPUTestSquare_ZeroDim(XPUTestSquare):
        def init_config(self):
            self.tmp_x = np.random.uniform(-2, 2, [])

    class XPUTestSquare2(XPUTestSquare):
        def init_config(self):
            self.tmp_x = np.random.uniform(-2, 2, [100])

    class XPUTestSquare3(XPUTestSquare):
        def init_config(self):
            self.tmp_x = np.random.uniform(-2, 2, [1, 15, 19])

    class XPUTestSquare4(XPUTestSquare):
        def init_config(self):
            self.tmp_x = np.random.uniform(-2, 2, [100, 10])

    class XPUTestSquare5(XPUTestSquare):
        def init_config(self):
            self.tmp_x = np.random.uniform(-2, 2, [1, 2, 5, 17])


support_types = get_xpu_op_support_types('square')
for stype in support_types:
    create_test_class(globals(), XPUTestSquareOP, stype)


class XPUTestPowOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'pow'
        self.use_dynamic_create_class = False

    class XPUTestPowBase(TestActivationOPBase):
        def set_case(self):
            self.dtype = self.in_type
            self.op_type = "pow"
            self.place = paddle.XPUPlace(0)
            self.inputs = {}
            self.init_config()
            self.init_data()
            self.attrs = {'factor': self.factor, 'use_xpu': True}
            if self.dtype == np.uint16:
                x_float32 = convert_uint16_to_float(self.inputs['X'])
                self.outputs = {'Out': np.power(x_float32, self.factor)}
            else:
                self.outputs = {'Out': np.power(self.inputs['X'], self.factor)}

        def init_config(self):
            self.range = (-1, 2)
            self.shape = [100]
            self.factor = 3.0

        def init_data(self):
            if self.dtype == np.uint16:
                x_float32 = np.random.uniform(
                    self.range[0], self.range[1], self.shape
                ).astype('float32')
                x = convert_float_to_uint16(x_float32)
                self.inputs = {'X': x}
            else:
                self.inputs = {
                    'X': np.random.uniform(
                        self.range[0], self.range[1], self.shape
                    ).astype(self.dtype)
                }

    class XPUTestPow1(XPUTestPowBase):
        def init_config(self):
            self.range = (-1, 1)
            self.shape = [1024, 8]
            self.factor = 1

    class XPUTestPow2(XPUTestPowBase):
        def init_config(self):
            self.range = (-1, 1)
            self.shape = [1024, 8]
            self.factor = 2

    class XPUTestPow3(XPUTestPowBase):
        def init_config(self):
            self.range = (-2, 2)
            self.shape = [4, 512, 15, 15]
            self.factor = 3

    class XPUTestPow4(XPUTestPowBase):
        def init_config(self):
            self.range = (-2, 2)
            self.shape = [4, 256, 22, 22]
            self.factor = 4

    class XPUTestPow5(XPUTestPowBase):
        def init_config(self):
            self.range = (0, 1)
            self.shape = [4, 256, 22, 22]
            self.factor = 1.2

    class XPUTestPow6(XPUTestPowBase):
        def init_config(self):
            self.range = (0, 1)
            self.shape = [1024, 8]
            self.factor = 3.2


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
                1,
            )
            out = leaky_relu(x, alpha)

            self.inputs = {'X': x}
            self.outputs = {'Out': out}
            self.attrs = {'use_xpu': True, 'alpha': alpha}


support_types = get_xpu_op_support_types('leaky_relu')
for stype in support_types:
    create_test_class(globals(), XPUTestLeakyReluOP, stype)


def leaky_relu(x, alpha):
    if alpha < 1:
        y_ref = np.maximum(x, alpha * x)
    else:
        y_ref = np.minimum(x, alpha * x)
    return y_ref.astype(x.dtype)


class XPUTestReciprocalOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'reciprocal'
        self.use_dynamic_create_class = False

    class XPUTestReciprocal(TestActivationOPBase):
        def set_case(self):
            self.op_type = "reciprocal"
            self.dtype = self.in_type

            np.random.seed(1024)
            x = np.random.uniform(1, 2, [1111, 1117]).astype(self.dtype)
            out = np.reciprocal(x)

            self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
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

    class XPUTestSoftPlus_ZeroDim(XPUTestSoftPlusBase):
        def init_config(self):
            self.x = np.random.uniform(-2, 2, []).astype(self.dtype)

    class XPUTestSoftPlus2(XPUTestSoftPlusBase):
        def init_config(self):
            self.x = np.random.uniform(-2, 2, [1024, 8]).astype(self.dtype)

    class XPUTestSoftPlus3(XPUTestSoftPlusBase):
        def init_config(self):
            self.x = np.random.uniform(-2, 2, [4, 512, 15, 15]).astype(
                self.dtype
            )

    class XPUTestSoftPlus4(XPUTestSoftPlusBase):
        def init_config(self):
            self.x = np.random.uniform(-2, 2, [4, 256, 22, 22]).astype(
                self.dtype
            )


support_types = get_xpu_op_support_types('softplus')
for stype in support_types:
    create_test_class(globals(), XPUTestSoftPlusOP, stype)


def ref_softplus(x, beta=1, threshold=20):
    x_beta = beta * x
    out = np.select(
        [x_beta <= threshold, x_beta > threshold],
        [np.log(1 + np.exp(x_beta)) / beta, x],
    )
    return out


# XPU_KP unittests, these ops can be found from xpu_op_kpfirst_list.h
class XPUTestBReluOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'brelu'
        self.use_dynamic_create_class = False

    class XPUTestBRelu(TestActivationOPBase):
        def set_case(self):
            self.op_type = "brelu"
            self.dtype = self.in_type

            np.random.seed(1024)
            x = np.random.uniform(-5, 10, [10, 12]).astype(self.dtype)
            t_min = 1.0
            t_max = 4.0
            # The same with TestAbs
            x[np.abs(x - t_min) < 0.005] = t_min + 0.02
            x[np.abs(x - t_max) < 0.005] = t_max + 0.02
            t = np.copy(x)
            t[t < t_min] = t_min
            t[t > t_max] = t_max

            self.inputs = {'X': x}
            self.outputs = {'Out': t}
            self.attrs = {'use_xpu': True, 't_min': t_min, 't_max': t_max}


support_types = get_xpu_op_support_types('brelu')
for stype in support_types:
    create_test_class(globals(), XPUTestBReluOP, stype)


class XPUTestCeilOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'ceil'
        self.use_dynamic_create_class = False

    class XPUTestCeil(TestActivationOPBase):
        def set_case(self):
            self.op_type = "ceil"
            self.dtype = self.in_type

            np.random.seed(1024)
            x = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype)
            out = np.ceil(x)

            self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
            self.outputs = {'Out': out}
            self.attrs = {'use_xpu': True}


support_types = get_xpu_op_support_types('ceil')
for stype in support_types:
    create_test_class(globals(), XPUTestCeilOP, stype)


class XPUTestCeluOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'celu'
        self.use_dynamic_create_class = False

    class XPUTestCelu(TestActivationOPBase):
        def set_case(self):
            self.op_type = "celu"
            self.dtype = self.in_type

            alpha = 1.5
            x = np.random.uniform(-3, 3, [10, 12]).astype(self.dtype)
            out = ref_celu(x, alpha)

            self.inputs = {'X': x}
            self.outputs = {'Out': out}
            self.attrs = {'use_xpu': True, 'alpha': alpha}


support_types = get_xpu_op_support_types('celu')
for stype in support_types:
    create_test_class(globals(), XPUTestCeluOP, stype)


def ref_celu(x, alpha):
    out_ref = np.maximum(0, x) + np.minimum(0, alpha * (np.exp(x / alpha) - 1))
    return out_ref.astype(x.dtype)


class XPUTestEluOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'elu'
        self.use_dynamic_create_class = False

    class XPUTestElu(TestActivationOPBase):
        def set_case(self):
            self.op_type = "elu"
            self.dtype = self.in_type

            alpha = 1.0
            x = np.random.uniform(-3, 3, [10, 12]).astype(self.dtype)
            out = ref_elu(x, alpha)

            self.inputs = {'X': x}
            self.outputs = {'Out': out}
            self.attrs = {'use_xpu': True, 'alpha': alpha}


support_types = get_xpu_op_support_types('elu')
for stype in support_types:
    create_test_class(globals(), XPUTestEluOP, stype)


def ref_elu(x, alpha):
    out_ref = np.where(x > 0, x, alpha * (np.exp(x) - 1))
    return out_ref.astype(x.dtype)


class XPUTestFloorOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'floor'
        self.use_dynamic_create_class = False

    class XPUTestFloor(TestActivationOPBase):
        def set_case(self):
            self.op_type = "floor"
            self.dtype = self.in_type

            np.random.seed(1024)
            x = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype)
            out = np.floor(x)

            self.inputs = {'X': x}
            self.outputs = {'Out': out}
            self.attrs = {'use_xpu': True}


support_types = get_xpu_op_support_types('floor')
for stype in support_types:
    create_test_class(globals(), XPUTestFloorOP, stype)


class XPUTestHardShrinkOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'hard_shrink'
        self.use_dynamic_create_class = False

    class XPUTestHardShrink(TestActivationOPBase):
        def set_case(self):
            self.op_type = "hard_shrink"
            self.dtype = self.in_type

            threshold = 0.5
            # self.set_attrs()
            np.random.seed(1024)
            x = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype) * 10
            out = ref_hardshrink(x, threshold)

            self.attrs = {'use_xpu': True}
            self.inputs = {'X': x}
            self.outputs = {'Out': out}


support_types = get_xpu_op_support_types('hard_shrink')
for stype in support_types:
    create_test_class(globals(), XPUTestHardShrinkOP, stype)


def ref_hardshrink(x, threshold):
    out = np.copy(x)
    out[(out >= -threshold) & (out <= threshold)] = 0
    return out


class XPUTestHardSigmoidOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'hard_sigmoid'
        self.use_dynamic_create_class = False

    class XPUTestHardSigmoid(TestActivationOPBase):
        def set_case(self):
            self.op_type = "hard_sigmoid"
            self.dtype = self.in_type
            self.slope = 0.166666666666667
            self.offset = 0.5

            x = np.random.uniform(-5, 5, [10, 12]).astype(self.dtype)
            lower_threshold = -self.offset / self.slope
            upper_threshold = (1.0 - self.offset) / self.slope

            # Same reason as TestAbs
            delta = 0.005
            x[np.abs(x - lower_threshold) < delta] = lower_threshold - 0.02
            x[np.abs(x - upper_threshold) < delta] = upper_threshold - 0.02

            out = ref_hardsigmoid(x, self.slope, self.offset)

            self.attrs = {
                'use_xpu': True,
                'slope': self.slope,
                'offset': self.offset,
            }
            self.inputs = {'X': x}
            self.outputs = {'Out': out}


support_types = get_xpu_op_support_types('hard_sigmoid')
for stype in support_types:
    create_test_class(globals(), XPUTestHardSigmoidOP, stype)


def ref_hardsigmoid(x, slope=0.166666666666667, offset=0.5):
    return np.maximum(np.minimum(x * slope + offset, 1.0), 0.0).astype(x.dtype)


class XPUTestLog1pOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'log1p'
        self.use_dynamic_create_class = False

    class XPUTestLog1p(TestActivationOPBase):
        def set_case(self):
            self.op_type = "log1p"
            self.dtype = self.in_type

            np.random.seed(1024)
            x = np.random.uniform(0.1, 1, [11, 17]).astype(self.dtype)
            out = np.log1p(x)

            self.inputs = {'X': x}
            self.outputs = {'Out': out}
            self.attrs = {'use_xpu': True}


support_types = get_xpu_op_support_types('log1p')
for stype in support_types:
    create_test_class(globals(), XPUTestLog1pOP, stype)


class XPUTestLogsigmoidOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'logsigmoid'
        self.use_dynamic_create_class = False

    class XPUTestLogsigmoid(TestActivationOPBase):
        def set_case(self):
            self.op_type = "logsigmoid"
            self.dtype = self.in_type

            np.random.seed(2048)
            x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
            out = np.log(1 / (1 + np.exp(-x)))

            self.inputs = {'X': x}
            self.outputs = {'Out': out}
            self.attrs = {'use_xpu': True}


support_types = get_xpu_op_support_types('logsigmoid')
for stype in support_types:
    create_test_class(globals(), XPUTestLogsigmoidOP, stype)


class XPUTestRelu6OP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'relu6'
        self.use_dynamic_create_class = False

    class XPUTestRelu6(TestActivationOPBase):
        def set_case(self):
            self.op_type = "relu6"
            self.dtype = self.in_type

            np.random.seed(1024)
            x = np.random.uniform(-1, 10, [10, 12]).astype(self.dtype)
            x[np.abs(x) < 0.005] = 0.02
            out = ref_relu6(x)

            self.attrs = {'use_xpu': True}
            self.inputs = {'X': x}
            self.outputs = {'Out': out}


support_types = get_xpu_op_support_types('relu6')
for stype in support_types:
    create_test_class(globals(), XPUTestRelu6OP, stype)


def ref_relu6(x, threshold=6.0):
    out = np.copy(x)
    out[np.abs(x - threshold) < 0.005] = threshold + 0.02
    out = np.minimum(np.maximum(x, 0), threshold)
    return out


class XPUTestSiluOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'silu'
        self.use_dynamic_create_class = False

    class XPUTestSilu(TestActivationOPBase):
        def set_case(self):
            self.op_type = "silu"
            self.dtype = self.in_type

            np.random.seed(1024)
            x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
            out = x / (np.exp(-x) + 1)

            self.inputs = {'X': x}
            self.outputs = {'Out': out}
            self.attrs = {'use_xpu': True}


support_types = get_xpu_op_support_types('silu')
for stype in support_types:
    create_test_class(globals(), XPUTestSiluOP, stype)


class XPUTestSoftReluOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'soft_relu'
        self.use_dynamic_create_class = False

    class XPUTestSoftRelu(TestActivationOPBase):
        def set_case(self):
            self.op_type = "soft_relu"
            self.dtype = self.in_type

            np.random.seed(4096)
            x = np.random.uniform(-3, 3, [4, 4]).astype(self.dtype)
            threshold = 2.0
            # The same reason with TestAbs
            x[np.abs(x - threshold) < 0.005] = threshold + 0.02
            x[np.abs(x + threshold) < 0.005] = -threshold - 0.02
            t = np.copy(x)
            t[t < -threshold] = -threshold
            t[t > threshold] = threshold
            out = np.log(np.exp(t) + 1)

            self.inputs = {'X': x}
            self.outputs = {'Out': out}
            self.attrs = {'use_xpu': True, 'threshold': threshold}


support_types = get_xpu_op_support_types('soft_relu')
for stype in support_types:
    create_test_class(globals(), XPUTestSoftReluOP, stype)


class XPUTestSoftSignOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'softsign'
        self.use_dynamic_create_class = False

    class XPUTestSoftSign(TestActivationOPBase):
        def set_case(self):
            self.op_type = "softsign"
            self.dtype = self.in_type

            np.random.seed(1024)
            x = np.random.uniform(-1, 1, [10, 12]).astype(self.dtype)
            out = ref_softsign(x)

            self.inputs = {'X': x}
            self.outputs = {'Out': out}
            self.attrs = {'use_xpu': True}


support_types = get_xpu_op_support_types('softsign')
for stype in support_types:
    create_test_class(globals(), XPUTestSoftSignOP, stype)


def ref_softsign(x):
    out = np.divide(x, 1 + np.abs(x))
    return out


class XPUTestSoftshrinkOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'softshrink'
        self.use_dynamic_create_class = False

    class XPUTestSoftshrink(TestActivationOPBase):
        def set_case(self):
            self.op_type = "softshrink"
            self.dtype = self.in_type

            threshold = 0.5
            np.random.seed(1023)
            x = np.random.uniform(0.25, 10, [10, 12]).astype(self.dtype)
            out = ref_softshrink(x, threshold)

            self.inputs = {'X': x}
            self.outputs = {'Out': out}
            self.attrs = {'use_xpu': True}


support_types = get_xpu_op_support_types('softshrink')
for stype in support_types:
    create_test_class(globals(), XPUTestSoftshrinkOP, stype)


def ref_softshrink(x, threshold=0.5):
    out = np.copy(x)
    out = (out < -threshold) * (out + threshold) + (out > threshold) * (
        out - threshold
    )
    return out


class XPUTestSwishOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'swish'
        self.use_dynamic_create_class = False

    class XPUTestSwishBase(TestActivationOPBase):
        def set_case(self):
            self.op_type = "swish"
            self.dtype = self.in_type

            self.init_config()
            out = ref_swish(self.x)

            if self.dtype == np.uint16:
                # bfloat16 actually
                new_x = convert_float_to_uint16(self.x)
            else:
                new_x = self.x.astype(self.dtype)

            self.inputs = {'X': new_x}
            self.outputs = {'Out': out}
            self.attrs = {'use_xpu': True}

        def init_config(self):
            self.x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)

    class XPUTestSwish_ZeroDim(XPUTestSwishBase):
        def init_config(self):
            self.x = np.random.uniform(-2, 2, []).astype(self.dtype)

    class XPUTestSwish2(XPUTestSwishBase):
        def init_config(self):
            self.x = np.random.uniform(-2, 2, [1024, 8]).astype(self.dtype)

    class XPUTestSwish3(XPUTestSwishBase):
        def init_config(self):
            self.x = np.random.uniform(-2, 2, [4, 512, 15, 15]).astype(
                self.dtype
            )

    class XPUTestSwish4(XPUTestSwishBase):
        def init_config(self):
            self.x = np.random.uniform(-2, 2, [4, 256, 22, 22]).astype(
                self.dtype
            )


support_types = get_xpu_op_support_types('swish')
for stype in support_types:
    create_test_class(globals(), XPUTestSwishOP, stype)


def ref_swish(x):
    from scipy.special import expit

    out = x * expit(x)
    return out


class XPUTestThresholdedReluOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'thresholded_relu'
        self.use_dynamic_create_class = False

    class XPUTestThresholdedRelu(TestActivationOPBase):
        def set_case(self):
            self.op_type = "thresholded_relu"
            self.dtype = self.in_type

            threshold = 1.0
            np.random.seed(1024)
            x = np.random.uniform(-20, 20, [10, 12]).astype(self.dtype)
            x[np.abs(x) < 0.005] = 0.02
            out = ref_thresholded_relu(x, threshold)

            self.inputs = {'X': x}
            self.outputs = {'Out': out}
            self.attrs = {'use_xpu': True}


support_types = get_xpu_op_support_types('thresholded_relu')
for stype in support_types:
    create_test_class(globals(), XPUTestThresholdedReluOP, stype)


def ref_thresholded_relu(x, threshold=1.0):
    out = (x > threshold) * x
    return out


class XPUTestMishOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'mish'
        self.use_dynamic_create_class = False

    class XPUTestMishBase(TestActivationOPBase):
        def set_case(self):
            self.op_type = "mish"
            self.dtype = self.in_type

            self.init_config()
            threshold = np.random.uniform(0, 1)
            out = ref_mish(self.x, threshold)

            self.inputs = {'X': self.x}
            self.outputs = {'Out': out}
            self.attrs = {'use_xpu': True, 'threshold': threshold}

        def init_config(self):
            self.x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)

    class XPUTestMish_ZeroDim(XPUTestMishBase):
        def init_config(self):
            self.x = np.random.uniform(-2, 2, []).astype(self.dtype)

    class XPUTestMish2(XPUTestMishBase):
        def init_config(self):
            self.x = np.random.uniform(-2, 2, [1024, 8]).astype(self.dtype)

    class XPUTestMish3(XPUTestMishBase):
        def init_config(self):
            self.x = np.random.uniform(-2, 2, [4, 512, 15, 15]).astype(
                self.dtype
            )

    class XPUTestMish4(XPUTestMishBase):
        def init_config(self):
            self.x = np.random.uniform(-2, 2, [4, 256, 22, 22]).astype(
                self.dtype
            )


support_types = get_xpu_op_support_types('mish')
for stype in support_types:
    create_test_class(globals(), XPUTestMishOP, stype)


def ref_mish(x, threshold=20):
    sp = np.select([x <= threshold, x > threshold], [np.log(1 + np.exp(x)), x])
    out = x * np.tanh(sp)
    return out


class XPUTestSinOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'sin'
        self.use_dynamic_create_class = False

    class XPUTestSinBase(TestActivationOPBase):
        def set_case(self):
            self.op_type = "sin"
            self.dtype = self.in_type

            self.init_config()
            if self.dtype == np.uint16:
                # bfloat16 actually
                self.x = convert_float_to_uint16(self.tmp_x)
            else:
                self.x = self.tmp_x.astype(self.dtype)
            out = np.sin(self.x)

            self.inputs = {'X': self.x}
            self.outputs = {'Out': out}
            self.attrs = {'use_xpu': True}

        def init_config(self):
            self.tmp_x = np.random.uniform(-np.pi, np.pi, [11, 17])

    class XPUTestSin_ZeroDim(XPUTestSinBase):
        def init_config(self):
            self.tmp_x = np.random.uniform(-np.pi, np.pi, [])

    class XPUTestSin2(XPUTestSinBase):
        def init_config(self):
            self.tmp_x = np.random.uniform(-np.pi, np.pi, [1024, 8])

    class XPUTestSin3(XPUTestSinBase):
        def init_config(self):
            self.tmp_x = np.random.uniform(-np.pi, np.pi, [4, 512, 15, 15])

    class XPUTestSin4(XPUTestSinBase):
        def init_config(self):
            self.tmp_x = np.random.uniform(-np.pi, np.pi, [4, 256, 22, 22])

    @check_run_big_shape_test()
    class XPUTestSinLargeShape1(XPUTestSinBase):
        def init_config(self):
            self.tmp_x = np.random.uniform(-np.pi, np.pi, [1, 8192, 1, 128])


support_types = get_xpu_op_support_types('sin')
for stype in support_types:
    create_test_class(globals(), XPUTestSinOP, stype)


class XPUTestCosOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = 'cos'
        self.use_dynamic_create_class = False

    class XPUTestCosBase(TestActivationOPBase):
        def set_case(self):
            self.op_type = "cos"
            self.dtype = self.in_type

            self.init_config()
            if self.dtype == np.uint16:
                # bfloat16 actually
                self.x = convert_float_to_uint16(self.tmp_x)
            else:
                self.x = self.tmp_x.astype(self.dtype)
            out = np.cos(self.x)

            self.inputs = {'X': self.x}
            self.outputs = {'Out': out}
            self.attrs = {'use_xpu': True}

        def init_config(self):
            self.tmp_x = np.random.uniform(-np.pi, np.pi, [11, 17])

    class XPUTestCos_ZeroDim(XPUTestCosBase):
        def init_config(self):
            self.tmp_x = np.random.uniform(-np.pi, np.pi, [])

    class XPUTestCos2(XPUTestCosBase):
        def init_config(self):
            self.tmp_x = np.random.uniform(-np.pi, np.pi, [1024, 8])

    class XPUTestCos3(XPUTestCosBase):
        def init_config(self):
            self.tmp_x = np.random.uniform(-np.pi, np.pi, [4, 512, 15, 15])

    class XPUTestCos4(XPUTestCosBase):
        def init_config(self):
            self.tmp_x = np.random.uniform(-np.pi, np.pi, [4, 256, 22, 22])

    @check_run_big_shape_test()
    class XPUTestCosLargeShape1(XPUTestCosBase):
        def init_config(self):
            self.tmp_x = np.random.uniform(-np.pi, np.pi, [1, 8192, 1, 128])


support_types = get_xpu_op_support_types('cos')
for stype in support_types:
    create_test_class(globals(), XPUTestCosOP, stype)


class XPUTestRsqrtOP(XPUOpTestWrapper):
    def __init__(self):
        self.op_name = "rsqrt"
        self.use_dynamic_create_class = False

    class XPUTestRsqrtBase(TestActivationOPBase):
        def set_case(self):
            self.dtype = self.in_type
            self.op_type = "rsqrt"
            self.place = paddle.XPUPlace(0)
            self.inputs = {}
            self.init_shape()
            self.init_data()
            if self.dtype == np.uint16:
                x_float32 = convert_uint16_to_float(self.inputs['X'])
                self.outputs = {'Out': np.reciprocal(np.sqrt(x_float32))}
            else:
                self.outputs = {'Out': np.reciprocal(np.sqrt(self.inputs['X']))}

        def init_shape(self):
            self.shape = (4, 10, 10)

        def init_data(self):
            if self.dtype == np.uint16:
                x = np.random.uniform(0.25, 1, self.shape).astype("float32")
                x = convert_float_to_uint16(x)
                self.inputs = {'X': x}
            else:
                self.inputs = {
                    'X': np.random.uniform(0.25, 1, self.shape).astype(
                        self.dtype
                    )
                }

    class TestRsqrtOp1(XPUTestRsqrtBase):
        def init_shape(self):
            self.shape = (8, 16, 8)

    class TestRsqrtOp2(XPUTestRsqrtBase):
        def init_shape(self):
            self.shape = (8, 16)

    class TestRsqrtOp3(XPUTestRsqrtBase):
        def init_shape(self):
            self.shape = (4, 8, 16)

    class TestRsqrtOp4(XPUTestRsqrtBase):
        def init_shape(self):
            self.shape = (4, 8, 8)

    class TestRsqrtOp5(XPUTestRsqrtBase):
        def init_shape(self):
            self.shape = (4, 8, 16)

    class TestRsqrtOp6(XPUTestRsqrtBase):
        def init_shape(self):
            self.shape = (1024, 8)

    class TestRsqrtOp7(XPUTestRsqrtBase):
        def init_shape(self):
            self.shape = (4, 512, 15, 15)

    class TestRsqrtOp8(XPUTestRsqrtBase):
        def init_shape(self):
            self.shape = (4, 256, 22, 22)

    class TestRsqrtOp_ZeorDim(XPUTestRsqrtBase):
        def init_shape(self):
            self.shape = []


support_types = get_xpu_op_support_types('rsqrt')
for stype in support_types:
    create_test_class(globals(), XPUTestRsqrtOP, stype)


if __name__ == "__main__":
    unittest.main()
