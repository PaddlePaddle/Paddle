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
from op_test import get_device_place, is_custom_device

import paddle


@unittest.skipIf(
    not (paddle.core.is_compiled_with_cuda() or is_custom_device()),
    "core is not compiled with CUDA",
)
class TestUnaryElementwiseOp_Stride(unittest.TestCase):
    def setUp(self):
        self.place = get_device_place()
        self.dtype = np.float64
        self.init_api()
        self.init_input()

    def init_api(self):
        self.paddle_api = paddle.cos
        self.numpy_api = np.cos

    def init_input(self):
        self.strided_input_type = "transpose"
        self.x = np.random.uniform(0.1, 1, [13, 17]).astype(self.dtype)
        self.perm = [1, 0]
        self.x_trans = np.transpose(self.x, self.perm)

    def test_dygraph_api_arithmetic(self):
        paddle.disable_static()
        x_trans = paddle.to_tensor(self.x_trans)
        if self.strided_input_type == "transpose":
            x_non_conti = paddle.transpose(x_trans, self.perm)
        elif self.strided_input_type == "as_stride":
            x_non_conti = paddle.as_strided(
                x_trans, self.shape_param, self.stride_param
            )
        else:
            raise TypeError(f"Unsupported test type {self.strided_input_type}.")
        out = self.paddle_api(x_non_conti)
        out_ref = self.numpy_api(self.x)
        np.testing.assert_allclose(out_ref, out.numpy())
        paddle.enable_static()


def create_test_act_stride_class(base_class, api_name, paddle_api, numpy_api):
    class TestStride1(base_class):
        def init_api(self):
            self.paddle_api = paddle_api
            self.numpy_api = numpy_api

        def init_input(self):
            self.strided_input_type = "transpose"
            self.x = np.random.randint(0, 256, [20, 2, 13, 17]).astype(
                self.dtype
            )
            self.perm = [0, 1, 3, 2]
            self.x_trans = np.transpose(self.x, self.perm)

    cls_name = "{}_{}_{}".format(base_class.__name__, api_name, "Stride1")
    TestStride1.__name__ = cls_name
    globals()[cls_name] = TestStride1

    class TestStride2(base_class):
        def init_input(self):
            self.strided_input_type = "transpose"
            self.x = np.random.uniform(0.1, 1, [20, 2, 13, 17]).astype(
                self.dtype
            )
            self.perm = [0, 2, 1, 3]
            self.x_trans = np.transpose(self.x, self.perm)

    cls_name = "{}_{}_{}".format(base_class.__name__, api_name, "Stride2")
    TestStride2.__name__ = cls_name
    globals()[cls_name] = TestStride2

    class TestStride3(base_class):
        def init_input(self):
            self.strided_input_type = "transpose"
            self.x = np.random.uniform(0.1, 1, [20, 2, 13, 17]).astype(
                self.dtype
            )
            self.perm = [0, 1, 3, 2]
            self.x_trans = np.transpose(self.x, self.perm)

    cls_name = "{}_{}_{}".format(base_class.__name__, api_name, "Stride3")
    TestStride3.__name__ = cls_name
    globals()[cls_name] = TestStride3

    class TestStride4(base_class):
        def init_input(self):
            self.strided_input_type = "transpose"
            self.x = np.random.uniform(0.1, 1, [1, 2, 13, 17]).astype(
                self.dtype
            )
            self.perm = [1, 0, 2, 3]
            self.x_trans = np.transpose(self.x, self.perm)

    cls_name = "{}_{}_{}".format(base_class.__name__, api_name, "Stride4")
    TestStride4.__name__ = cls_name
    globals()[cls_name] = TestStride4

    class TestStride5(base_class):
        def init_input(self):
            self.strided_input_type = "as_stride"
            self.x = np.random.uniform(0.1, 1, [23, 2, 13, 20]).astype(
                self.dtype
            )
            self.x_trans = self.x
            self.x = self.x[:, 0:1, :, 0:1]
            self.shape_param = [23, 1, 13, 1]
            self.stride_param = [520, 260, 20, 1]

    cls_name = "{}_{}_{}".format(base_class.__name__, api_name, "Stride5")
    TestStride5.__name__ = cls_name
    globals()[cls_name] = TestStride5

    class TestStrideZeroDim1(base_class):
        def init_input(self):
            self.strided_input_type = "transpose"
            self.x = np.random.uniform(0.1, 1, []).astype(self.dtype)
            self.perm = []
            self.x_trans = np.transpose(self.x, self.perm)

    cls_name = "{}_{}_{}".format(
        base_class.__name__, api_name, "StrideZeroDim1"
    )
    TestStrideZeroDim1.__name__ = cls_name
    globals()[cls_name] = TestStrideZeroDim1

    class TestStrideZeroSize1(base_class):
        def init_input(self):
            self.strided_input_type = "transpose"
            self.x = np.random.rand(1, 0, 2).astype('float32')
            self.perm = [2, 1, 0]
            self.x_trans = np.transpose(self.x, self.perm)

    cls_name = "{}_{}_{}".format(
        base_class.__name__, api_name, "StrideZeroSize1"
    )
    TestStrideZeroSize1.__name__ = cls_name
    globals()[cls_name] = TestStrideZeroSize1


create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Cos", paddle.cos, np.cos
)
create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Sin", paddle.sin, np.sin
)
create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Tan", paddle.tan, np.tan
)
create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Acos", paddle.acos, np.arccos
)
create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Asin", paddle.asin, np.arcsin
)
create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Atan", paddle.atan, np.arctan
)
create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Sinh", paddle.sinh, np.sinh
)
create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Cosh", paddle.cosh, np.cosh
)
create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Tanh", paddle.tanh, np.tanh
)
create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Asinh", paddle.asinh, np.arcsinh
)
create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Acosh", paddle.acosh, np.arccosh
)
create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Atanh", paddle.atanh, np.arctanh
)
create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Square", paddle.square, np.square
)
create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Sqrt", paddle.sqrt, np.sqrt
)


def rsqrt_ref(x):
    out = 1.0 / np.sqrt(x)
    return out


create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Rsqrt", paddle.rsqrt, rsqrt_ref
)
create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride,
    "Reciprocal",
    paddle.reciprocal,
    np.reciprocal,
)
create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Floor", paddle.floor, np.floor
)
create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Ceil", paddle.ceil, np.ceil
)
create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Log", paddle.log, np.log
)
create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Log2", paddle.log2, np.log2
)
create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Log10", paddle.log10, np.log10
)
create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Log1p", paddle.log1p, np.log1p
)
create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Exp", paddle.exp, np.exp
)
create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Log1p", paddle.expm1, np.expm1
)
create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Round", paddle.round, np.round
)
create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Abs", paddle.abs, np.abs
)


def relu_ref(x):
    out = np.maximum(x, 0)
    return out


create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Relu", paddle.nn.functional.relu, relu_ref
)


def silu_ref(x_np):
    out = x_np / (1 + np.exp(-x_np))
    return out


create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Silu", paddle.nn.functional.silu, silu_ref
)


def ref_sigmoid(x):
    out = 1 / (1 + np.exp(-x))
    return out


create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride,
    "Sigmoid",
    paddle.nn.functional.sigmoid,
    ref_sigmoid,
)


def ref_log_sigmoid(x):
    out = np.log(1 / (1 + np.exp(-x)))
    return out


create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride,
    "LogSigmoid",
    paddle.nn.functional.log_sigmoid,
    ref_log_sigmoid,
)


def ref_softsign(x):
    out = np.divide(x, 1 + np.abs(x))
    return out


create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride,
    "Softsign",
    paddle.nn.functional.softsign,
    ref_softsign,
)


def ref_leaky_relu(x, alpha=0.01):
    out = np.copy(x)
    out[out < 0] *= alpha
    return out


create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride,
    "LeakyRelu",
    paddle.nn.functional.leaky_relu,
    ref_leaky_relu,
)


def ref_hardshrink_v2(x, threshold=0.5):
    out = np.copy(x)
    out[(out >= -threshold) & (out <= threshold)] = 0
    return out


create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride,
    "Hardshrink",
    paddle.nn.functional.hardshrink,
    ref_hardshrink_v2,
)


def ref_softshrink(x, threshold=0.5):
    out = np.copy(x)
    out = (out < -threshold) * (out + threshold) + (out > threshold) * (
        out - threshold
    )
    return out


create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride,
    "Softshrink",
    paddle.nn.functional.softshrink,
    ref_softshrink,
)


def ref_elu(x, alpha=1):
    out_ref = np.where(x > 0, x, alpha * (np.exp(x) - 1))
    return out_ref.astype(x.dtype)


create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Elu", paddle.nn.functional.elu, ref_elu
)


def ref_celu(x, alpha=1):
    out_ref = np.maximum(0, x) + np.minimum(0, alpha * (np.exp(x / alpha) - 1))
    return out_ref.astype(x.dtype)


create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Celu", paddle.nn.functional.celu, ref_celu
)


def ref_mish(x, threshold=20.0):
    softplus = np.select(
        [x <= threshold, x > threshold], [np.log(1 + np.exp(x)), x]
    )
    return x * np.tanh(softplus)


create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride, "Mish", paddle.nn.functional.mish, ref_mish
)


def ref_hardtanh(x, min=-1.0, max=1.0):
    out = np.copy(x)
    out[np.abs(x - min) < 0.005] = min + 0.02
    out[np.abs(x - max) < 0.005] = max + 0.02
    out = np.minimum(np.maximum(x, min), max)
    return out


create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride,
    "Hardtanh",
    paddle.nn.functional.hardtanh,
    ref_hardtanh,
)


def ref_softplus(x, beta=1, threshold=20):
    x_beta = beta * x
    out = np.select(
        [x_beta <= threshold, x_beta > threshold],
        [np.log(1 + np.exp(x_beta)) / beta, x],
    )
    return out


create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride,
    "Softplus",
    paddle.nn.functional.softplus,
    ref_softplus,
)


def ref_hardsigmoid(x, slope=0.166666666666667, offset=0.5):
    return np.maximum(np.minimum(x * slope + offset, 1.0), 0.0).astype(x.dtype)


create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride,
    "Hardsigmoid",
    paddle.nn.functional.hardsigmoid,
    ref_hardsigmoid,
)


def ref_selu(
    x,
    scale=1.0507009873554804934193349852946,
    alpha=1.6732632423543772848170429916717,
):
    out = np.copy(x)
    out_flat = out.flatten()
    for i in range(out_flat.size):
        if out_flat[i] < 0:
            out_flat[i] = alpha * np.exp(out_flat[i]) - alpha
        out_flat[i] = scale * out_flat[i]
    out = out_flat.reshape(x.shape)
    return out


create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride,
    "Hardtanh",
    paddle.nn.functional.selu,
    ref_selu,
)


def ref_hardswish(x, threshold=6.0, scale=6.0, offset=3.0):
    x_dtype = x.dtype
    if x_dtype == 'float16':
        x_dtype = 'float16'
        x = x.astype('float32')
    return (
        x * np.minimum(np.maximum(x + offset, 0.0), threshold) / scale
    ).astype(x_dtype)


create_test_act_stride_class(
    TestUnaryElementwiseOp_Stride,
    "Hardswish",
    paddle.nn.functional.hardswish,
    ref_hardswish,
)

if __name__ == "__main__":
    unittest.main()
