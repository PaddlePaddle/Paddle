#!/usr/bin/env python3

# Copyright (c) 2023 CINN Authors. All Rights Reserved.
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
from cinn.common import Bool, Float, Int, is_compiled_with_cuda
from cinn.frontend import NetBuilder
from op_test import OpTest, OpTestTool

import paddle


def cinn_dtype_convert(dtype_str):
    if dtype_str == "float32":
        return Float(32)
    elif dtype_str == "int64":
        return Int(64)
    elif dtype_str == "bool":
        return Bool()
    else:
        print("Datatype %s has not been supported yet", dtype_str)


##################################
#     TestElementwiseAddGrad     #
##################################
# 1) x is 0D, y is 0D
@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestElementwiseAddGrad(OpTest):
    def setUp(self):
        np.random.seed(2023)
        self.init_input()

    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype("float32"),
            "y": np.random.randint(-10, 10, []).astype("float32"),
            "dout": np.random.randint(-10, 10, []).astype("float32"),
        }

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        y = paddle.to_tensor(self.inputs["y"], stop_gradient=False)
        out = paddle.add(x, y)

        self.paddle_outputs = [out]
        self.paddle_grads = self.get_paddle_grads(
            [out], [x, y], [self.inputs["dout"]]
        )

    def build_cinn_program(self, target):
        builder = NetBuilder("add")
        x = builder.create_input(Float(32), self.inputs["x"].shape, "x")
        y = builder.create_input(Float(32), self.inputs["y"].shape, "y")
        # Test elementwise_add here, next unittest tests add, actually these two APIs are same.
        out = builder.elementwise_add(x, y)

        dout = builder.create_input(
            Float(32), self.inputs["dout"].shape, "dout"
        )
        x_grad, y_grad = builder.elementwise_add_grad(dout, x, y)

        prog = builder.build()
        res = self.get_cinn_output(
            prog,
            target,
            [x, y, dout],
            [self.inputs["x"], self.inputs["y"], self.inputs["dout"]],
            [out, x_grad, y_grad],
        )

        out, x_grad, y_grad = res
        self.cinn_outputs = [out]
        self.cinn_grads = [x_grad, y_grad]
        self.assertEqual(out.shape, self.inputs["dout"].shape)
        self.assertEqual(x_grad.shape, self.inputs["x"].shape)
        self.assertEqual(y_grad.shape, self.inputs["y"].shape)

    def test_check_results(self):
        self.check_outputs_and_grads()


# 2) x is ND, y is 0D
# NOTE: CINN only supports x's rank >= y's rank, hence no need to test next scenario: `3) x is 0D, y is ND`
class TestElementwiseAddGrad1(TestElementwiseAddGrad):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, [3, 5]).astype("float32"),
            "y": np.random.randint(-10, 10, []).astype("float32"),
            "dout": np.random.randint(-10, 10, [3, 5]).astype("float32"),
        }


##################################
#    TestElementwiseBinaryOp     #
##################################
@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestElementwiseBinaryOp_0DTo0D(OpTest):
    def setUp(self):
        np.random.seed(2023)
        self.init_dtype()
        self.init_input()

    def init_dtype(self):
        self.dtype = "float32"

    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
            "y": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.target_shape = ()

    def paddle_func(self, x, y):
        return paddle.add(x, y)

    def cinn_func(self, builder, x, y):
        return builder.add(x, y)

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        y = paddle.to_tensor(self.inputs["y"], stop_gradient=False)
        out = self.paddle_func(x, y)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("binary_op")
        x = builder.create_input(
            cinn_dtype_convert(self.dtype), self.inputs["x"].shape, "x"
        )
        y = builder.create_input(
            cinn_dtype_convert(self.dtype), self.inputs["y"].shape, "y"
        )
        out = self.cinn_func(builder, x, y)

        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x, y], [self.inputs["x"], self.inputs["y"]], [out]
        )

        self.cinn_outputs = res
        self.assertEqual(res[0].shape, self.target_shape)

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestElementwiseBinaryOp_NdTo0d(TestElementwiseBinaryOp_0DTo0D):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, [3, 5]).astype(self.dtype),
            "y": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.target_shape = (3, 5)


def create_unit_test(
    parent, test_name, fn_paddle, fn_cinn, dtype="float32", **kwargs
):
    @OpTestTool.skip_if(
        not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
    )
    class TestClass(parent):
        def setUp(self):
            super().setUp()
            for k, v in kwargs.items():
                setattr(self, k, v)

        def init_dtype(self):
            self.dtype = dtype

        def paddle_func(self, *args):
            return fn_paddle(*args)

        def cinn_func(self, builder, *args):
            return eval(fn_cinn)(*args)

    cls_name = f"{parent.__name__}_{test_name}"
    TestClass.__name__ = cls_name
    globals()[cls_name] = TestClass


# NOTE: CINN only supports x's rank >= y's rank, hence no need to test scenario: x is 0D, y is ND
create_unit_test(
    TestElementwiseBinaryOp_0DTo0D, "sub", paddle.subtract, "builder.subtract"
)
create_unit_test(
    TestElementwiseBinaryOp_NdTo0d, "sub", paddle.subtract, "builder.subtract"
)
create_unit_test(
    TestElementwiseBinaryOp_0DTo0D, "mul1", paddle.multiply, "builder.multiply"
)
create_unit_test(
    TestElementwiseBinaryOp_NdTo0d, "mul1", paddle.multiply, "builder.multiply"
)
create_unit_test(
    TestElementwiseBinaryOp_0DTo0D,
    "mul2",
    paddle.multiply,
    "builder.elementwise_mul",
)
create_unit_test(
    TestElementwiseBinaryOp_NdTo0d,
    "mul2",
    paddle.multiply,
    "builder.elementwise_mul",
)
create_unit_test(
    TestElementwiseBinaryOp_0DTo0D, "div", paddle.divide, "builder.divide"
)
create_unit_test(
    TestElementwiseBinaryOp_NdTo0d, "div", paddle.divide, "builder.divide"
)
# Paddle'atan2 only supports 0D + 0D -> 0D
create_unit_test(
    TestElementwiseBinaryOp_0DTo0D, "atan2", paddle.atan2, "builder.atan2"
)
create_unit_test(
    TestElementwiseBinaryOp_0DTo0D,
    "floor_divide",
    paddle.floor_divide,
    "builder.floor_divide",
    dtype="int64",
)
create_unit_test(
    TestElementwiseBinaryOp_NdTo0d,
    "floor_divide",
    paddle.floor_divide,
    "builder.floor_divide",
    dtype="int64",
)
create_unit_test(
    TestElementwiseBinaryOp_0DTo0D,
    "mod",
    paddle.mod,
    "builder.mod",
    dtype="int64",
)
create_unit_test(
    TestElementwiseBinaryOp_NdTo0d,
    "mod",
    paddle.mod,
    "builder.mod",
    dtype="int64",
)
create_unit_test(
    TestElementwiseBinaryOp_0DTo0D,
    "remainder",
    paddle.remainder,
    "builder.remainder",
    dtype="int64",
)
create_unit_test(
    TestElementwiseBinaryOp_NdTo0d,
    "remainder",
    paddle.remainder,
    "builder.remainder",
    dtype="int64",
)
create_unit_test(
    TestElementwiseBinaryOp_0DTo0D, "max", paddle.maximum, "builder.max"
)
create_unit_test(
    TestElementwiseBinaryOp_NdTo0d, "max", paddle.maximum, "builder.max"
)
create_unit_test(
    TestElementwiseBinaryOp_0DTo0D, "min", paddle.minimum, "builder.min"
)
create_unit_test(
    TestElementwiseBinaryOp_NdTo0d, "min", paddle.minimum, "builder.min"
)
create_unit_test(
    TestElementwiseBinaryOp_0DTo0D, "pow", paddle.pow, "builder.pow"
)
create_unit_test(
    TestElementwiseBinaryOp_NdTo0d, "pow", paddle.pow, "builder.pow"
)

create_unit_test(
    TestElementwiseBinaryOp_0DTo0D,
    "logical_and",
    paddle.logical_and,
    "builder.logical_and",
    dtype="bool",
)
create_unit_test(
    TestElementwiseBinaryOp_NdTo0d,
    "logical_and",
    paddle.logical_and,
    "builder.logical_and",
    dtype="bool",
)
create_unit_test(
    TestElementwiseBinaryOp_0DTo0D,
    "logical_or",
    paddle.logical_or,
    "builder.logical_or",
    dtype="bool",
)
create_unit_test(
    TestElementwiseBinaryOp_NdTo0d,
    "logical_or",
    paddle.logical_or,
    "builder.logical_or",
    dtype="bool",
)
create_unit_test(
    TestElementwiseBinaryOp_0DTo0D,
    "logical_xor",
    paddle.logical_xor,
    "builder.logical_xor",
    dtype="bool",
)
create_unit_test(
    TestElementwiseBinaryOp_NdTo0d,
    "logical_xor",
    paddle.logical_xor,
    "builder.logical_xor",
    dtype="bool",
)

create_unit_test(
    TestElementwiseBinaryOp_0DTo0D,
    "bitwise_and",
    paddle.bitwise_and,
    "builder.bitwise_and",
    dtype="int64",
)
create_unit_test(
    TestElementwiseBinaryOp_NdTo0d,
    "bitwise_and",
    paddle.bitwise_and,
    "builder.bitwise_and",
    dtype="int64",
)
create_unit_test(
    TestElementwiseBinaryOp_0DTo0D,
    "bitwise_or",
    paddle.bitwise_or,
    "builder.bitwise_or",
    dtype="int64",
)
create_unit_test(
    TestElementwiseBinaryOp_NdTo0d,
    "bitwise_or",
    paddle.bitwise_or,
    "builder.bitwise_or",
    dtype="int64",
)
create_unit_test(
    TestElementwiseBinaryOp_0DTo0D,
    "bitwise_xor",
    paddle.bitwise_xor,
    "builder.bitwise_xor",
    dtype="int64",
)
create_unit_test(
    TestElementwiseBinaryOp_NdTo0d,
    "bitwise_xor",
    paddle.bitwise_xor,
    "builder.bitwise_xor",
    dtype="int64",
)

create_unit_test(
    TestElementwiseBinaryOp_0DTo0D,
    "equal",
    paddle.equal,
    "builder.equal",
    dtype="int64",
)
create_unit_test(
    TestElementwiseBinaryOp_NdTo0d,
    "equal",
    paddle.equal,
    "builder.equal",
    dtype="int64",
)
create_unit_test(
    TestElementwiseBinaryOp_0DTo0D,
    "not_equal",
    paddle.not_equal,
    "builder.not_equal",
    dtype="int64",
)
create_unit_test(
    TestElementwiseBinaryOp_NdTo0d,
    "not_equal",
    paddle.not_equal,
    "builder.not_equal",
    dtype="int64",
)
create_unit_test(
    TestElementwiseBinaryOp_0DTo0D,
    "greater_than",
    paddle.greater_than,
    "builder.greater_than",
    dtype="int64",
)
create_unit_test(
    TestElementwiseBinaryOp_NdTo0d,
    "greater_than",
    paddle.greater_than,
    "builder.greater_than",
    dtype="int64",
)
create_unit_test(
    TestElementwiseBinaryOp_0DTo0D,
    "greater_equal",
    paddle.greater_equal,
    "builder.greater_equal",
    dtype="int64",
)
create_unit_test(
    TestElementwiseBinaryOp_NdTo0d,
    "greater_equal",
    paddle.greater_equal,
    "builder.greater_equal",
    dtype="int64",
)
create_unit_test(
    TestElementwiseBinaryOp_0DTo0D,
    "less_than",
    paddle.less_than,
    "builder.less_than",
    dtype="int64",
)
create_unit_test(
    TestElementwiseBinaryOp_NdTo0d,
    "less_than",
    paddle.less_than,
    "builder.less_than",
    dtype="int64",
)
create_unit_test(
    TestElementwiseBinaryOp_0DTo0D,
    "less_equal",
    paddle.less_equal,
    "builder.less_equal",
    dtype="int64",
)
create_unit_test(
    TestElementwiseBinaryOp_NdTo0d,
    "less_equal",
    paddle.less_equal,
    "builder.less_equal",
    dtype="int64",
)
create_unit_test(
    TestElementwiseBinaryOp_0DTo0D,
    "is_close",
    paddle.isclose,
    "builder.isclose",
)


######################
#    TestUnaryOp     #
######################
@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestUnaryOp(OpTest):
    def setUp(self):
        np.random.seed(2023)
        self.init_dtype()
        self.init_input()

    def init_dtype(self):
        self.dtype = "float32"

    def init_input(self):
        self.inputs = {
            "x": np.random.uniform(0.0, 1.0, []).astype(self.dtype),
        }
        self.target_shape = ()

    def paddle_func(self, x):
        return paddle.sqrt(x)

    def cinn_func(self, builder, x):
        return builder.sqrt(x)

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        out = self.paddle_func(x)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("unary_op")
        x = builder.create_input(
            cinn_dtype_convert(self.dtype), self.inputs["x"].shape, "x"
        )
        out = self.cinn_func(builder, x)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])

        self.cinn_outputs = res
        self.assertEqual(res[0].shape, self.target_shape)

    def test_check_results(self):
        self.check_outputs_and_grads()


create_unit_test(TestUnaryOp, "tanh", paddle.tanh, "builder.tanh")
create_unit_test(TestUnaryOp, "relu", paddle.nn.functional.relu, "builder.relu")
create_unit_test(
    TestUnaryOp, "relu6", paddle.nn.functional.relu6, "builder.relu6"
)
create_unit_test(TestUnaryOp, "gelu", paddle.nn.functional.gelu, "builder.gelu")
create_unit_test(
    TestUnaryOp, "sigmoid", paddle.nn.functional.sigmoid, "builder.sigmoid"
)
create_unit_test(TestUnaryOp, "exp", paddle.exp, "builder.exp")
create_unit_test(TestUnaryOp, "erf", paddle.erf, "builder.erf")
create_unit_test(TestUnaryOp, "rsqrt", paddle.rsqrt, "builder.rsqrt")
create_unit_test(TestUnaryOp, "log", paddle.log, "builder.log")
create_unit_test(TestUnaryOp, "log2", paddle.log2, "builder.log2")
create_unit_test(TestUnaryOp, "log10", paddle.log10, "builder.log10")
create_unit_test(TestUnaryOp, "floor", paddle.floor, "builder.floor")
create_unit_test(TestUnaryOp, "ceil", paddle.ceil, "builder.ceil")
create_unit_test(TestUnaryOp, "round", paddle.round, "builder.round")
create_unit_test(TestUnaryOp, "trunc", paddle.trunc, "builder.trunc")
create_unit_test(TestUnaryOp, "sin", paddle.sin, "builder.sin")
create_unit_test(TestUnaryOp, "cos", paddle.cos, "builder.cos")
create_unit_test(TestUnaryOp, "tan", paddle.tan, "builder.tan")
create_unit_test(TestUnaryOp, "sinh", paddle.sinh, "builder.sinh")
create_unit_test(TestUnaryOp, "cosh", paddle.cosh, "builder.cosh")
create_unit_test(TestUnaryOp, "asin", paddle.asin, "builder.asin")
create_unit_test(TestUnaryOp, "acos", paddle.acos, "builder.acos")
create_unit_test(TestUnaryOp, "atan", paddle.atan, "builder.atan")
create_unit_test(TestUnaryOp, "asinh", paddle.asinh, "builder.asinh")
create_unit_test(TestUnaryOp, "atanh", paddle.atanh, "builder.atanh")
create_unit_test(TestUnaryOp, "isnan", paddle.isnan, "builder.is_nan")
create_unit_test(TestUnaryOp, "isfinite", paddle.isfinite, "builder.is_finite")
create_unit_test(TestUnaryOp, "isinf", paddle.isinf, "builder.is_inf")
create_unit_test(
    TestUnaryOp,
    "logical_not",
    paddle.logical_not,
    "builder.logical_not",
    dtype="bool",
)
create_unit_test(
    TestUnaryOp,
    "bitwise_not",
    paddle.bitwise_not,
    "builder.bitwise_not",
    dtype="int64",
)
create_unit_test(TestUnaryOp, "negative", paddle.neg, "builder.negative")
create_unit_test(TestUnaryOp, "sign", paddle.sign, "builder.sign")
create_unit_test(TestUnaryOp, "abs", paddle.abs, "builder.abs")
create_unit_test(
    TestUnaryOp, "reciprocal", paddle.reciprocal, "builder.reciprocal"
)
create_unit_test(
    TestUnaryOp, "softmax", paddle.nn.functional.softmax, "builder.softmax"
)


# acosh requires input value > 1.0, specific init_input instead of using create_unit_test
class TestUnaryOp_acosh(TestUnaryOp):
    def init_input(self):
        self.inputs = {
            "x": np.random.uniform(1.0, 10.0, []).astype(self.dtype),
        }
        self.target_shape = ()

    def paddle_func(self, x):
        return paddle.acosh(x)

    def cinn_func(self, builder, x):
        return builder.acosh(x)


#######################
#    TestSundryOp     #
#######################
@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestScaleOp(OpTest):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.init_input()

    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.target_shape = ()

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        out = paddle.scale(x, scale=2.0, bias=1.0)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("scale_op")
        x = builder.create_input(
            cinn_dtype_convert(self.dtype), self.inputs["x"].shape, "x"
        )
        out = builder.scale(x, 2.0, 1.0)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])

        self.cinn_outputs = res
        self.assertEqual(res[0].shape, self.target_shape)

    def test_check_results(self):
        self.check_outputs_and_grads()


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestCastOp(OpTest):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.init_input()

    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.target_shape = ()

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        out = paddle.cast(x, 'int32')

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("cast_op")
        x = builder.create_input(
            cinn_dtype_convert(self.dtype), self.inputs["x"].shape, "x"
        )
        out = builder.cast(x, "int32")

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])

        self.cinn_outputs = res
        self.assertEqual(res[0].shape, self.target_shape)

    def test_check_results(self):
        self.check_outputs_and_grads()


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestArgmaxOp(OpTest):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.init_input()

    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.param = (0,)
        self.target_shape = ()

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        out = paddle.argmax(x, *self.param)
        out = paddle.cast(out, 'int32')

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("argmax_op")
        x = builder.create_input(
            cinn_dtype_convert(self.dtype), self.inputs["x"].shape, "x"
        )
        out = builder.argmax(x, *self.param)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])

        self.cinn_outputs = res
        self.assertEqual(res[0].shape, self.target_shape)

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestArgmaxOp2(TestArgmaxOp):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.param = (-1,)
        self.target_shape = ()


class TestArgmaxOp1D(TestArgmaxOp):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, [5]).astype(self.dtype),
        }
        self.param = (0,)
        self.target_shape = ()


class TestArgmaxOp2D(TestArgmaxOp):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, [3, 5]).astype(self.dtype),
        }
        self.param = (0,)
        self.target_shape = (5,)


class TestArgmaxOp2DKeepDim(TestArgmaxOp):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, [3, 5]).astype(self.dtype),
        }
        self.param = (0, True)
        self.target_shape = (1, 5)


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestArgminOp(OpTest):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.init_input()

    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.param = (0,)
        self.target_shape = ()

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        out = paddle.argmin(x, *self.param)
        out = paddle.cast(out, 'int32')

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("argmin_op")
        x = builder.create_input(
            cinn_dtype_convert(self.dtype), self.inputs["x"].shape, "x"
        )
        out = builder.argmin(x, *self.param)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])

        self.cinn_outputs = res
        self.assertEqual(res[0].shape, self.target_shape)

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestArgminOp2(TestArgminOp):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.param = (-1,)
        self.target_shape = ()


class TestArgminOp1D(TestArgminOp):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, [5]).astype(self.dtype),
        }
        self.param = (0,)
        self.target_shape = ()


class TestArgminOp2D(TestArgminOp):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, [3, 5]).astype(self.dtype),
        }
        self.param = (0,)
        self.target_shape = (5,)


class TestArgminOp2DKeepDim(TestArgminOp):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, [3, 5]).astype(self.dtype),
        }
        self.param = (0, True)
        self.target_shape = (1, 5)


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestTransposeOp(OpTest):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.init_input()

    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.target_shape = ()

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        out = paddle.transpose(x, [])

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("transpose_op")
        x = builder.create_input(
            cinn_dtype_convert(self.dtype), self.inputs["x"].shape, "x"
        )
        out = builder.transpose(x, [])

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])

        self.cinn_outputs = res
        self.assertEqual(res[0].shape, self.target_shape)

    def test_check_results(self):
        self.check_outputs_and_grads()


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestArgsortOp(OpTest):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.init_input()

    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.axis = -1
        self.target_shape = ()

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        out = paddle.argsort(x, axis=self.axis)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("argsort_op")
        x = builder.create_input(
            cinn_dtype_convert(self.dtype), self.inputs["x"].shape, "x"
        )
        out = builder.argsort(x, self.axis, True)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], out)

        self.cinn_outputs = np.array([res[0]]).astype("int64")
        self.assertEqual(res[0].shape, self.target_shape)

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestArgsortOp2(TestArgsortOp):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.axis = 0
        self.target_shape = ()


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestSortOp(OpTest):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.init_input()

    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.axis = -1
        self.target_shape = ()

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        out = paddle.sort(x, axis=self.axis)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("sort_op")
        x = builder.create_input(
            cinn_dtype_convert(self.dtype), self.inputs["x"].shape, "x"
        )
        out = builder.sort(x, self.axis, True)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])

        self.cinn_outputs = res
        self.assertEqual(res[0].shape, self.target_shape)

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestSortOp2(TestSortOp):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.axis = 0
        self.target_shape = ()


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestTopkOp(OpTest):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.init_input()

    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.axis = -1
        self.target_shape = ()

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        out, indices = paddle.topk(x, k=1, axis=self.axis)

        self.paddle_outputs = [out, indices]

    def build_cinn_program(self, target):
        builder = NetBuilder("topk_op")
        x = builder.create_input(
            cinn_dtype_convert(self.dtype), self.inputs["x"].shape, "x"
        )
        out = builder.top_k(x, 1, self.axis, True)

        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x], [self.inputs["x"]], [out[0], out[1]]
        )

        self.cinn_outputs = res
        self.assertEqual(res[0].shape, self.target_shape)
        self.assertEqual(res[1].shape, self.target_shape)

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestTopkOp2(TestTopkOp):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.axis = 0
        self.target_shape = ()


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestExpandDimsOp(OpTest):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.init_input()

    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.unsqueeze_dim = [0]
        self.target_shape = (1,)

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        out = paddle.unsqueeze(x, self.unsqueeze_dim)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("unsqueeze_op")
        x = builder.create_input(
            cinn_dtype_convert(self.dtype), self.inputs["x"].shape, "x"
        )
        out = builder.expand_dims(x, self.unsqueeze_dim)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])

        self.cinn_outputs = res
        self.assertEqual(res[0].shape, self.target_shape)

    def test_check_results(self):
        self.check_outputs_and_grads()


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestExpandDimsOp2D(TestExpandDimsOp):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.unsqueeze_dim = [0, 1]
        self.target_shape = (
            1,
            1,
        )


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestBroadcastToOp1D(OpTest):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.init_input()

    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.broadcast_shape = [1]

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        out = paddle.broadcast_to(x, shape=self.broadcast_shape)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("broadcast_to_op")
        x = builder.create_input(
            cinn_dtype_convert(self.dtype), self.inputs["x"].shape, "x"
        )
        out = builder.broadcast_to(x, self.broadcast_shape)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])

        self.cinn_outputs = res
        self.assertEqual(list(res[0].shape), list(self.broadcast_shape))

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestBroadcastToOp2D(TestBroadcastToOp1D):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.broadcast_shape = [1, 1]


class TestBroadcastToOp3D(TestBroadcastToOp1D):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.broadcast_shape = [3, 3, 3]


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestReverseOp(OpTest):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.init_input()

    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.target_shape = ()

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        out = paddle.reverse(x, axis=[])

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("reverse_op")
        x = builder.create_input(
            cinn_dtype_convert(self.dtype), self.inputs["x"].shape, "x"
        )
        out = builder.reverse(x, [])

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])

        self.cinn_outputs = res
        self.assertEqual(res[0].shape, self.target_shape)

    def test_check_results(self):
        self.check_outputs_and_grads()


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestSumOp(OpTest):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.init_input()

    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
            "y": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.target_shape = ()

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        y = paddle.to_tensor(self.inputs["y"], stop_gradient=False)
        out = x + y

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("sum_op")
        x = builder.create_input(
            cinn_dtype_convert(self.dtype), self.inputs["x"].shape, "x"
        )
        y = builder.create_input(
            cinn_dtype_convert(self.dtype), self.inputs["y"].shape, "y"
        )
        out = builder.sum([x, y])

        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x, y], [self.inputs["x"], self.inputs["y"]], [out]
        )

        self.cinn_outputs = res
        self.assertEqual(res[0].shape, self.target_shape)

    def test_check_results(self):
        self.check_outputs_and_grads()


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestDropoutOp(OpTest):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.init_input()

    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.target_shape = ()

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        out = paddle.nn.functional.dropout(x, 1.0)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("dropout_op")
        x = builder.create_input(
            cinn_dtype_convert(self.dtype), self.inputs["x"].shape, "x"
        )
        out = builder.dropout_infer(x, 1.0)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])

        self.cinn_outputs = res
        self.assertEqual(res[0].shape, self.target_shape)

    def test_check_results(self):
        self.check_outputs_and_grads()


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestReshapeOp(OpTest):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.init_input()

    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.target_shape = [1]

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        out = paddle.reshape(x, self.target_shape)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("reshape_op")
        x = builder.create_input(
            cinn_dtype_convert(self.dtype), self.inputs["x"].shape, "x"
        )
        out = builder.reshape(x, self.target_shape)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])

        self.cinn_outputs = res
        self.assertEqual(list(res[0].shape), [1] * len(self.target_shape))

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestReshapeOp0DTo2D(TestReshapeOp):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.target_shape = [1, 1]


class TestReshapeOp0DTo1D_DS(TestReshapeOp):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.target_shape = [-1]


class TestReshapeOp0DTo2D_DS(TestReshapeOp):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.target_shape = [-1, 1]


class TestReshapeOp0DTo0D(TestReshapeOp):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.target_shape = []


class TestReshapeOp1DTo0D(TestReshapeOp):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, [1]).astype(self.dtype),
        }
        self.target_shape = []


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestFillConstantOp(OpTest):
    def setUp(self):
        np.random.seed(2023)
        self.target_shape = ()

    def build_paddle_program(self, target):
        out = paddle.full([], 123.456, "float32")

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("fill_constant_op")
        out = builder.fill_constant([], 123.456, "out", "float32")

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [], [], [out])

        self.cinn_outputs = res
        self.assertEqual(res[0].shape, self.target_shape)

    def test_check_results(self):
        self.check_outputs_and_grads()


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestSqueezeOp(OpTest):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.init_input()

    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.squeeze_axex = [0]
        self.target_shape = ()

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        out = paddle.squeeze(x, axis=self.squeeze_axex)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("squeeze_op")
        x = builder.create_input(
            cinn_dtype_convert(self.dtype), self.inputs["x"].shape, "x"
        )
        out = builder.squeeze(x, self.squeeze_axex)

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])

        self.cinn_outputs = res
        self.assertEqual(res[0].shape, self.target_shape)

    def test_check_results(self):
        self.check_outputs_and_grads()


class TestSqueezeOp1D(TestSqueezeOp):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, [1]).astype(self.dtype),
        }
        self.squeeze_axex = []
        self.target_shape = ()


class TestSqueezeOp2D(TestSqueezeOp):
    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, [1, 1]).astype(self.dtype),
        }
        self.squeeze_axex = [0, 1]
        self.target_shape = ()


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestGaussianRandomOp(OpTest):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.target_shape = ()

    def build_paddle_program(self, target):
        out = paddle.tensor.random.gaussian(
            shape=[],
            mean=0.0,
            std=0.0,
            dtype=self.dtype,
        )
        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("gaussian_random_op")

        out = builder.gaussian_random(
            [],
            0.0,
            0.0,
            1234,
            self.dtype,
        )

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [], [], [out])
        self.cinn_outputs = res

        self.assertEqual(res[0].shape, self.target_shape)

    def test_check_results(self):
        self.check_outputs_and_grads()


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestMatmulOp(OpTest):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.init_input()

    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, [10]).astype(self.dtype),
            "y": np.random.randint(-10, 10, [10]).astype(self.dtype),
        }
        self.target_shape = ()

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        y = paddle.to_tensor(self.inputs["y"], stop_gradient=False)
        out = paddle.matmul(x, y)

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("matmul_op")
        x = builder.create_input(
            cinn_dtype_convert(self.dtype), self.inputs["x"].shape, "x"
        )
        y = builder.create_input(
            cinn_dtype_convert(self.dtype), self.inputs["y"].shape, "y"
        )
        out = builder.matmul(x, y)

        prog = builder.build()
        res = self.get_cinn_output(
            prog, target, [x, y], [self.inputs["x"], self.inputs["y"]], [out]
        )

        self.cinn_outputs = res
        self.assertEqual(res[0].shape, self.target_shape)

    def test_check_results(self):
        self.check_outputs_and_grads()


@OpTestTool.skip_if(
    not is_compiled_with_cuda(), "x86 test will be skipped due to timeout."
)
class TestFlipOp(OpTest):
    def setUp(self):
        np.random.seed(2023)
        self.dtype = "float32"
        self.init_input()

    def init_input(self):
        self.inputs = {
            "x": np.random.randint(-10, 10, []).astype(self.dtype),
        }
        self.target_shape = ()

    def build_paddle_program(self, target):
        x = paddle.to_tensor(self.inputs["x"], stop_gradient=False)
        out = paddle.flip(x, axis=[])

        self.paddle_outputs = [out]

    def build_cinn_program(self, target):
        builder = NetBuilder("flip_op")
        x = builder.create_input(
            cinn_dtype_convert(self.dtype), self.inputs["x"].shape, "x"
        )
        out = builder.flip(x, [])

        prog = builder.build()
        res = self.get_cinn_output(prog, target, [x], [self.inputs["x"]], [out])

        self.cinn_outputs = res
        self.assertEqual(res[0].shape, self.target_shape)

    def test_check_results(self):
        self.check_outputs_and_grads()


if __name__ == "__main__":
    unittest.main()
