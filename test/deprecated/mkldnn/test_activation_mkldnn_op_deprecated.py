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

import sys
import unittest

import numpy as np

sys.path.append("../../mkldnn")
from mkldnn_op_test import check_if_mkldnn_primitives_exist_in_bwd
from op_test import OpTest, convert_float_to_uint16
from test_activation_op import (
    TestAbs,
    TestAbs_ZeroDim,
    TestActivation,
    TestActivation_ZeroDim,
    TestHardSwish,
    TestHardSwish_ZeroDim,
    TestLeakyRelu,
    TestLeakyRelu_ZeroDim,
    TestRelu,
    TestRelu6,
    TestRelu6_ZeroDim,
    TestRelu_ZeroDim,
    TestSigmoid,
    TestSigmoid_ZeroDim,
    TestSoftplus,
    TestSoftplus_ZeroDim,
    TestSqrt,
    TestSqrt_ZeroDim,
    TestSwish,
    TestSwish_ZeroDim,
    TestTanh,
    TestTanh_ZeroDim,
)
from test_gelu_op import gelu
from utils import compare_legacy_with_pt

import paddle
import paddle.nn.functional as F
from paddle.base import core


class TestMKLDNNReluDim2(TestRelu):
    def setUp(self):
        super().setUp()

        self.attrs = {"use_mkldnn": True}

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNRelu_ZeroDim(TestRelu_ZeroDim):
    def setUp(self):
        super().setUp()

        self.attrs = {"use_mkldnn": True}

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNRelu6Dim2(TestRelu6):
    def setUp(self):
        super().setUp()
        self.attrs.update({"use_mkldnn": True})
        self.check_pir_onednn = False

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNRelu6_ZeroDim(TestRelu6_ZeroDim):
    def setUp(self):
        super().setUp()
        self.attrs.update({"use_mkldnn": True})
        self.check_pir_onednn = False

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNLeakyReluDim2(TestLeakyRelu):
    def setUp(self):
        super().setUp()

        self.attrs = {"use_mkldnn": True}
        self.check_pir_onednn = False

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output(check_dygraph=False, check_pir_onednn=True)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'], 'Out', check_dygraph=False, check_pir_onednn=False
        )


class TestMKLDNNLeakyRelu_ZeroDim(TestLeakyRelu_ZeroDim):
    def setUp(self):
        super().setUp()

        self.attrs = {"use_mkldnn": True}
        self.check_pir_onednn = False

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNGeluDim2(TestActivation):
    def setUp(self):
        self.op_type = "gelu"
        self.python_api = F.gelu
        self.dtype = np.float32

        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = gelu(x, False)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True}
        self.check_pir_onednn = False


class TestMKLDNNGelu_ZeroDim(TestActivation_ZeroDim):
    def setUp(self):
        self.op_type = "gelu"
        self.python_api = F.gelu
        self.dtype = np.float32

        x = np.random.uniform(-1, 1, []).astype(self.dtype)
        out = gelu(x, False)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True}
        self.check_pir_onednn = False


class TestMKLDNNGeluDim2Approx(TestActivation):
    def setUp(self):
        self.op_type = "gelu"
        self.python_api = F.gelu
        self.dtype = np.float32

        x = np.random.uniform(-1, 1, [11, 17]).astype(self.dtype)
        out = gelu(x, True)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True, "approximate": True}
        self.check_pir_onednn = False


class TestMKLDNNTanhDim2(TestTanh):
    def setUp(self):
        super().setUp()

        self.attrs = {"use_mkldnn": True}
        self.check_pir_onednn = False

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNTanh_ZeroDim(TestTanh_ZeroDim):
    def setUp(self):
        super().setUp()

        self.attrs = {"use_mkldnn": True}
        self.check_pir_onednn = False

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNSqrtDim2(TestSqrt):
    def setUp(self):
        super().setUp()

        self.attrs = {"use_mkldnn": True}
        self.check_pir_onednn = False

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNSqrt_ZeroDim(TestSqrt_ZeroDim):
    def setUp(self):
        super().setUp()

        self.attrs = {"use_mkldnn": True}
        self.check_pir_onednn = False

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNAbsDim2(TestAbs):
    def setUp(self):
        super().setUp()
        self.attrs = {"use_mkldnn": True}

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNAbs_ZeroDim(TestAbs_ZeroDim):
    def setUp(self):
        super().setUp()
        self.attrs = {"use_mkldnn": True}

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNSwishDim2(TestSwish):
    def setUp(self):
        super().setUp()

        self.attrs["use_mkldnn"] = True
        self.check_pir_onednn = False

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNSwish_ZeroDim(TestSwish_ZeroDim):
    def setUp(self):
        super().setUp()

        self.attrs["use_mkldnn"] = True
        self.check_eager = False
        self.check_pir_onednn = False

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNHardSwishDim2(TestHardSwish):
    def setUp(self):
        super().setUp()
        self.attrs = {"use_mkldnn": True}
        self.check_pir_onednn = False


class TestMKLDNNHardSwish_ZeroDim(TestHardSwish_ZeroDim):
    def setUp(self):
        super().setUp()
        self.attrs = {"use_mkldnn": True}
        self.check_pir_onednn = False


class TestMKLDNNSigmoidDim2(TestSigmoid):
    def setUp(self):
        super().setUp()
        self.attrs = {"use_mkldnn": True}


class TestMKLDNNSigmoid_ZeroDim(TestSigmoid_ZeroDim):
    def setUp(self):
        super().setUp()
        self.attrs = {"use_mkldnn": True}


class TestMKLDNNReluDim4(TestRelu):
    def setUp(self):
        super().setUp()

        x = np.random.uniform(-1, 1, [2, 4, 3, 5]).astype("float32")
        # The same reason with TestAbs
        x[np.abs(x) < 0.005] = 0.02
        out = np.maximum(x, 0)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True}

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNLeakyReluDim4(TestLeakyRelu):
    def setUp(self):
        super().setUp()

        x = np.random.uniform(-1, 1, [2, 4, 3, 5]).astype("float32")
        # The same reason with TestAbs
        x[np.abs(x) < 0.005] = 0.02
        out = np.maximum(x, 0.02 * x)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True}
        self.check_pir_onednn = False

    def init_dtype(self):
        self.dtype = np.float32

    def test_check_output(self):
        self.check_output(check_dygraph=False, check_pir_onednn=True)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(
            ['X'], 'Out', check_dygraph=False, check_pir_onednn=False
        )


class TestMKLDNNGeluDim4(TestActivation):
    def setUp(self):
        self.op_type = "gelu"
        self.python_api = F.gelu
        self.dtype = np.float32

        x = np.random.uniform(-1, 1, [2, 4, 3, 5]).astype(self.dtype)
        out = gelu(x, False)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True}
        self.check_pir_onednn = False


class TestMKLDNNGeluDim4Approx(TestActivation):
    def setUp(self):
        self.op_type = "gelu"
        self.python_api = F.gelu
        self.dtype = np.float32

        x = np.random.uniform(-1, 1, [2, 4, 3, 5]).astype(self.dtype)
        out = gelu(x, True)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True, "approximate": True}
        self.check_pir_onednn = False


@unittest.skipIf(
    not core.supports_bfloat16(), "place does not support BF16 evaluation"
)
class TestMKLDNNGeluBf16Dim4(TestActivation):
    def setUp(self):
        self.op_type = "gelu"
        self.python_api = F.gelu
        self.dtype = np.uint16

        x = np.random.uniform(-1, 1, [2, 4, 3, 5]).astype(np.float32)
        out = convert_float_to_uint16(gelu(x, False))

        self.inputs = {'X': convert_float_to_uint16(x)}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True}
        self.check_pir_onednn = False

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace(), check_pir_onednn=True)

    def test_check_grad(self):
        pass


@unittest.skipIf(
    not core.supports_bfloat16(), "place does not support BF16 evaluation"
)
class TestMKLDNNGeluBf16Dim4Approx(TestActivation):
    def setUp(self):
        self.op_type = "gelu"
        self.python_api = F.gelu
        self.dtype = np.uint16

        x = np.random.uniform(-1, 1, [2, 4, 3, 5]).astype(np.float32)
        out = convert_float_to_uint16(gelu(x, True))

        self.inputs = {'X': convert_float_to_uint16(x)}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True, "approximate": True}
        self.check_pir_onednn = False

    def test_check_output(self):
        self.check_output_with_place(core.CPUPlace(), check_pir_onednn=True)

    def test_check_grad(self):
        pass


class TestMKLDNNTanhDim4(TestTanh):
    def setUp(self):
        super().setUp()

        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 4, 3, 5]).astype("float32")
        }
        self.outputs = {'Out': np.tanh(self.inputs['X'])}
        self.attrs = {"use_mkldnn": True}
        self.check_pir_onednn = False


class TestMKLDNNSqrtDim4(TestSqrt):
    def setUp(self):
        super().setUp()

        self.inputs = {
            'X': np.random.uniform(0.1, 1, [2, 4, 3, 5]).astype("float32")
        }
        self.outputs = {'Out': np.sqrt(self.inputs['X'])}
        self.attrs = {"use_mkldnn": True}
        self.check_pir_onednn = False


class TestMKLDNNAbsDim4(TestAbs):
    def setUp(self):
        super().setUp()

        x = np.random.uniform(-1, 1, [2, 4, 3, 5]).astype("float32")
        # The same reason with TestAbs
        x[np.abs(x) < 0.005] = 0.02
        self.inputs = {'X': x}
        self.outputs = {'Out': np.abs(self.inputs['X'])}
        self.attrs = {"use_mkldnn": True}

    def init_dtype(self):
        self.dtype = np.float32


def ref_hardswish(x, threshold=6.0, scale=6.0, offset=3.0):
    x_dtype = x.dtype
    if x_dtype == 'float16':
        x_dtype = 'float16'
        x = x.astype('float32')
    return (
        x * np.minimum(np.maximum(x + offset, 0.0), threshold) / scale
    ).astype(x_dtype)


class TestMKLDNNHardSwishDim4(TestHardSwish):
    def setUp(self):
        super().setUp()

        x = np.random.uniform(0.1, 1, [2, 4, 3, 5]).astype(self.dtype)
        threshold = 6.0
        scale = 6.0
        offset = 3.0
        x[np.abs(x + offset) < 0.005] = 0.02
        x[np.abs(x - threshold + offset) < 0.005] = threshold - offset + 0.02

        out = ref_hardswish(x, threshold, scale, offset)

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True}
        self.check_pir_onednn = False

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNMish(TestActivation):
    def setUp(self):
        self.op_type = "mish"
        self.python_api = F.mish
        self.dtype = np.float32

        x = np.random.uniform(0.1, 1, [2, 4, 3, 5]).astype(self.dtype)
        out = x * np.tanh(np.log(1 + np.exp(x)))

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True}
        self.check_pir_onednn = False


class TestMKLDNNMish_ZeroDim(TestActivation_ZeroDim):
    def setUp(self):
        self.op_type = "mish"
        self.python_api = F.mish
        self.dtype = np.float32

        x = np.random.uniform(0.1, 1, []).astype(self.dtype)
        out = x * np.tanh(np.log(1 + np.exp(x)))

        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True}
        self.check_pir_onednn = False


class TestMKLDNNRound(TestActivation):
    def setUp(self):
        self.op_type = "round"
        self.python_api = paddle.round
        x = np.random.uniform(0.1, 1, [2, 4, 3, 5]).astype(np.float32)
        out = np.round(x)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True}
        self.check_pir_onednn = False

    def test_check_output(self):
        self.check_output(check_pir=True, check_pir_onednn=True)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', check_pir=True, check_pir_onednn=False)


class TestMKLDNNRound_ZeroDim(TestActivation_ZeroDim):
    def setUp(self):
        self.op_type = "round"
        self.python_api = paddle.round
        x = np.random.uniform(0.1, 1, []).astype(np.float32)
        out = np.round(x)

        self.inputs = {'X': x}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True}
        self.check_pir_onednn = False

    def test_check_output(self):
        self.check_output(check_pir=True, check_pir_onednn=True)

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        self.check_grad(['X'], 'Out', check_pir=True, check_pir_onednn=False)


class TestMKLDNNSigmoidDim4(TestSigmoid):
    def setUp(self):
        super().setUp()

        x = np.random.uniform(0.1, 1, [2, 4, 3, 5]).astype(self.dtype)
        out = 1 / (1 + np.exp(-x))
        self.inputs = {'X': OpTest.np_dtype_to_base_dtype(x)}
        self.outputs = {'Out': out}
        self.attrs = {"use_mkldnn": True}


class TestMKLDNNEluDefaultAlpha(TestActivation):
    def setUp(self):
        self.op_type = "elu"
        self.python_api = F.elu
        self.set_alpha()

        x = np.random.random((5, 5, 4)).astype("float32")

        self.inputs = {'X': x}
        self.attrs = {'use_mkldnn': True, 'alpha': self.alpha}
        self.outputs = {
            'Out': np.maximum(0, x)
            + np.minimum(0, self.alpha * (np.exp(x) - 1))
        }
        self.check_pir_onednn = False

    def set_alpha(self):
        self.alpha = 1.0


class TestMKLDNNEluDefaultAlpha_ZeroDim(TestActivation_ZeroDim):
    def setUp(self):
        self.op_type = "elu"
        self.python_api = F.elu
        self.set_alpha()

        x = np.random.random(()).astype("float32")

        self.inputs = {'X': x}
        self.attrs = {'use_mkldnn': True, 'alpha': self.alpha}
        self.outputs = {
            'Out': np.maximum(0, x)
            + np.minimum(0, self.alpha * (np.exp(x) - 1))
        }
        self.check_pir_onednn = False

    def set_alpha(self):
        self.alpha = 1.0


class TestMKLDNNEluCustomAlpha(TestMKLDNNEluDefaultAlpha):
    def set_alpha(self):
        self.alpha = 2.5


class TestMKLDNNExpOp(TestActivation):
    def setUp(self):
        self.op_type = "exp"
        self.python_api = paddle.exp
        x = np.random.random((5, 5, 4)).astype("float32")

        self.inputs = {'X': x}
        self.attrs = {'use_mkldnn': True}
        self.outputs = {'Out': np.exp(x)}
        self.check_pir_onednn = False


class TestMKLDNNExpOp_ZeroDim(TestActivation_ZeroDim):
    def setUp(self):
        self.op_type = "exp"
        self.python_api = paddle.exp
        x = np.random.random(()).astype("float32")

        self.inputs = {'X': x}
        self.attrs = {'use_mkldnn': True}
        self.outputs = {'Out': np.exp(x)}
        self.check_pir_onednn = False


# Check if primitives already exist in backward
class TestMKLDNNAbsPrimitivesAlreadyExist(unittest.TestCase):
    def setUp(self):
        paddle.enable_static()
        super().setUp()

        np.random.seed(123)
        self.op_type = 'abs'
        self.python_api = paddle.abs
        self.x = np.random.uniform(-1, 1, [2, 2]).astype(np.float32)
        self.out = np.abs(self.x)
        self.out_grad = np.random.random_sample(self.x.shape).astype(np.float32)
        self.x_grad = self.__abs_bwd(self.x, self.out_grad)

    # Abs grad calculation
    def __abs_bwd(self, x, out_grad):
        return out_grad * np.sign(x)

    @compare_legacy_with_pt
    def test_check(self):
        check_if_mkldnn_primitives_exist_in_bwd(
            self, self.op_type, self.x, self.out, self.out_grad, self.x_grad
        )


class TestMKLDNNSoftplusDim2(TestSoftplus):
    def setUp(self):
        super().setUp()
        self.attrs.update({"use_mkldnn": True})
        self.check_pir_onednn = False

    def init_dtype(self):
        self.dtype = np.float32


class TestMKLDNNSoftplus_ZeroDim(TestSoftplus_ZeroDim):
    def setUp(self):
        super().setUp()
        self.attrs.update({"use_mkldnn": True})

    def init_dtype(self):
        self.dtype = np.float32


if __name__ == '__main__':
    unittest.main()
