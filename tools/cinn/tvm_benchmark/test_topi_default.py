# Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

import numpy
import tvm
import tvm.testing
from tvm import te, topi

dtype = ["float32", "float32", "float32", "float32"]
target = "llvm"
ctx = tvm.context(target, 0)
repeat = 10


def test_op(
    func, input_shapes, out_shape, attrs={}, name="test_op", dtype=dtype
):
    assert len(input_shapes) >= 1
    A = te.placeholder(input_shapes[0], name="A", dtype=dtype[0])
    if len(input_shapes) == 1:
        C = func(A)
    elif len(input_shapes) == 2:
        B = te.placeholder(input_shapes[1], name="B", dtype=dtype[1])
        C = func(A, B)
    elif len(input_shapes) == 3:
        B = te.placeholder(input_shapes[1], name="B", dtype=dtype[1])
        B1 = te.placeholder(input_shapes[2], name="B1", dtype=dtype[2])
        C = func(A, B, B1)
    # Default schedule
    s = te.create_schedule(C.op)
    if len(input_shapes) == 1:
        func = tvm.build(s, [A, C], target=target, name=name)
    elif len(input_shapes) == 2:
        func = tvm.build(s, [A, B, C], target=target, name=name)
    elif len(input_shapes) == 3:
        func = tvm.build(s, [A, B, B1, C], target=target, name=name)
    assert func
    print(func)
    a = tvm.nd.array(numpy.random.random(input_shapes[0]).astype(dtype[0]), ctx)
    if len(input_shapes) > 1:
        b = tvm.nd.array(
            numpy.random.random(input_shapes[1]).astype(dtype[1]), ctx
        )
    if len(input_shapes) > 2:
        b1 = tvm.nd.array(
            numpy.random.random(input_shapes[2]).astype(dtype[2]), ctx
        )
    c = tvm.nd.array(numpy.zeros(out_shape, dtype=dtype[len(dtype) - 1]), ctx)

    evaluator = func.time_evaluator(func.entry_name, ctx, number=repeat)
    print(f"repeat: {repeat:f}")
    if len(input_shapes) == 1:
        print("Baseline: %f" % (evaluator(a, c).mean * 1000))
        print(tvm.lower(s, [A, C], simple_mode=True))
    elif len(input_shapes) == 2:
        print("Baseline: %f" % (evaluator(a, b, c).mean * 1000))
        print(tvm.lower(s, [A, B, C], simple_mode=True))
    elif len(input_shapes) == 3:
        print("Baseline: %f" % (evaluator(a, b, b1, c).mean * 1000))
        print(tvm.lower(s, [A, B, B1, C], simple_mode=True))


def test_elementwise():
    input_shapes, out_shape = [(100, 32), (100, 32)], (100, 32)
    # input_shapes1, out_shape1 = [(1024, 1024, 1024),
    #                              (1024, 1024, 1024)], (1024, 1024, 1024)
    input_shapes2, out_shape2 = [(1024, 14, 14), (1024, 14, 14)], (1024, 14, 14)

    def compute_add(A, B):
        return topi.add(A, B)

    def compute_mul(A, B):
        return topi.multiply(A, B)

    test_op(compute_add, input_shapes, out_shape, name="elementwise_add")
    # test_op(compute_add, input_shapes1, out_shape1, name="elementwise_add")
    test_op(compute_add, input_shapes2, out_shape2, name="elementwise_add")
    test_op(compute_mul, input_shapes, out_shape, name="elementwise_mul")
    # test_op(compute_mul, input_shapes1, out_shape1, name="elementwise_mul")
    test_op(compute_mul, input_shapes2, out_shape2, name="elementwise_mul")


def test_relu():
    input_shapes, out_shape = [(2, 512, 7, 7)], (2, 512, 7, 7)
    input_shapes1, out_shape1 = [(1024, 1024, 1024)], (1024, 1024, 1024)
    input_shapes2, out_shape2 = [(1024, 14, 14)], (1024, 14, 14)
    input_shapes3, out_shape3 = [(100, 32)], (100, 32)
    name = "relu"

    def compute(A):
        return topi.nn.relu(A)

    test_op(compute, input_shapes, out_shape, name=name)
    test_op(compute, input_shapes1, out_shape1, name=name)
    test_op(compute, input_shapes2, out_shape2, name=name)
    test_op(compute, input_shapes3, out_shape3, name=name)


def test_conv2d_nchw():
    input_shapes, out_shape = [(2, 512, 7, 7), (512, 512, 3, 3)], (2, 512, 5, 5)
    name = "conv2d_nchw"
    strides, padding, dilation = [1, 1], [0, 0], [1, 1]

    def compute(A, B):
        return topi.nn.conv2d(
            A, B, strides, padding, dilation, layout="NCHW", out_dtype=None
        )

    test_op(compute, input_shapes, out_shape, name=name)


# depthwise_conv2d_nchw
def test_depthwise_conv2d_nchw():
    input_shapes, out_shape = [(2, 32, 112, 112), (32, 1, 3, 3)], (
        2,
        32,
        112,
        112,
    )
    name = "depthwise_conv2d_nchw"
    strides, padding, dilation = [1, 1], [1, 1], [1, 1]

    def compute(A, B):
        return topi.nn.depthwise_conv2d_nchw(
            A, B, strides, padding, dilation, out_dtype=None
        )

    test_op(compute, input_shapes, out_shape, name=name)


def test_pool2d():
    input_shapes, out_shape = [(2, 64, 112, 112)], (2, 64, 56, 56)
    name = "pool2d"
    kernel, stride, padding = [3, 3], [2, 2], [1, 1, 1, 1]
    pool_type = "max"

    def compute(A):
        return topi.nn.pool(
            A,
            kernel,
            stride,
            padding,
            pool_type,
            ceil_mode=False,
            layout="NCHW",
            count_include_pad=False,
        )

    test_op(compute, input_shapes, out_shape, name=name)


def test_softmax():
    input_shapes, out_shape = [(1024, 2048)], (1024, 2048)
    input_shapes1, out_shape1 = [(3, 1000)], (3, 1000)
    name = "softmax"

    def compute(A):
        return topi.nn.softmax(A)

    test_op(compute, input_shapes, out_shape, name=name)
    test_op(compute, input_shapes1, out_shape1, name=name)


def test_unary():
    input_shapes, out_shape = [(1024, 2048)], (1024, 2048)
    input_shapes1, out_shape1 = [(3, 1000)], (3, 1000)
    input_shapes2, out_shape2 = [(1024, 2047)], (1024, 2047)

    def test_unary_basic(name, func):
        def compute(A):
            return func(A)

        test_op(compute, input_shapes, out_shape, name=name)
        test_op(compute, input_shapes1, out_shape1, name=name)
        test_op(compute, input_shapes2, out_shape2, name=name)

    for opfunc in [
        topi.exp,
        topi.erf,
        topi.sigmoid,
        topi.sqrt,
        topi.log,
        topi.log2,
        topi.log10,
        topi.floor,
        topi.ceil,
        topi.round,
        topi.trunc,
        topi.cos,
        topi.cosh,
        topi.tan,
        topi.tanh,
        topi.sin,
        topi.sinh,
        topi.acos,
        topi.acosh,
        topi.asin,
        topi.asinh,
        topi.atan,
        topi.atanh,
    ]:
        test_unary_basic(str(opfunc), opfunc)


def test_is():
    input_shapes, out_shape = [(1024, 2048)], (1024, 2048)
    input_shapes1, out_shape1 = [(3, 1000)], (3, 1000)
    input_shapes2, out_shape2 = [(1024, 2047)], (1024, 2047)
    type = ["float32", "bool"]

    def test_is_basic(name, func):
        def compute(A):
            return func(A)

        test_op(compute, input_shapes, out_shape, name=name, dtype=type)
        test_op(compute, input_shapes1, out_shape1, name=name, dtype=type)
        test_op(compute, input_shapes2, out_shape2, name=name, dtype=type)

    for opfunc in [
        topi.isnan,
        topi.isfinite,
        topi.isinf,
    ]:
        test_is_basic(str(opfunc), opfunc)


def test_bitwise_not():
    input_shapes, out_shape = [(1024, 2048)], (1024, 2048)
    input_shapes1, out_shape1 = [(3, 1000)], (3, 1000)
    input_shapes2, out_shape2 = [(1024, 2047)], (1024, 2047)
    type = ["int32", "int32", "int32"]

    def test_unary_basic(name, func):
        def compute(A):
            return func(A)

        test_op(compute, input_shapes, out_shape, name=name, dtype=type)
        test_op(compute, input_shapes1, out_shape1, name=name, dtype=type)
        test_op(compute, input_shapes2, out_shape2, name=name, dtype=type)

    for opfunc in [
        topi.bitwise_not,
    ]:
        test_unary_basic(str(opfunc), opfunc)


def test_bitwise_binary():
    input_shapes, out_shape = [(1024, 2048), (1024, 2048)], (1024, 2048)
    input_shapes1, out_shape1 = [(3, 1000), (3, 1000)], (3, 1000)
    input_shapes2, out_shape2 = [(1024, 2047), (1024, 2047)], (1024, 2047)
    type = ["int32", "int32", "int32"]

    def test_binary_basic(name, func):
        def compute(A, B):
            return func(A, B)

        test_op(compute, input_shapes, out_shape, name=name, dtype=type)
        test_op(compute, input_shapes1, out_shape1, name=name, dtype=type)
        test_op(compute, input_shapes2, out_shape2, name=name, dtype=type)

    for opfunc in [
        topi.bitwise_or,
        topi.bitwise_and,
        topi.bitwise_xor,
        topi.left_shift,
        topi.right_shift,
    ]:
        test_binary_basic(str(opfunc), opfunc)


def test_sigmoid():
    input_shapes, out_shape = [(2, 672, 1, 1)], (2, 672, 1, 1)
    input_shapes1, out_shape1 = [(3, 1000)], (3, 1000)
    name = "sigmoid"

    def compute(A):
        return topi.sigmoid(A)

    test_op(compute, input_shapes, out_shape, name=name)
    test_op(compute, input_shapes1, out_shape1, name=name)


def test_matmul():
    input_shapes, out_shape = [(32, 32), (32, 32)], (32, 32)
    input_shapes1, out_shape1 = [(512, 512), (512, 512)], (512, 512)
    # input_shapes2, out_shape2 = [(1024,1024),(1024,1024)], (1024,1024)
    input_shapes3, out_shape3 = [(100, 32), (32, 100)], (100, 100)
    name = "matmul"

    def compute(A, B):
        return topi.matmul(A, B, False, False)

    test_op(compute, input_shapes, out_shape, name=name)
    test_op(compute, input_shapes1, out_shape1, name=name)
    # test_op(compute, input_shapes2, out_shape2, name=name)
    test_op(compute, input_shapes3, out_shape3, name=name)


# batch_norm
def test_batch_norm():
    input_shapes, out_shape = [(2, 32, 112, 112), (32,), (32,)], (
        2,
        32,
        112,
        112,
    )
    # mean,variance=32,32
    name = "batch_norm"

    def compute(A, Scale, Shift):
        return te.compute(
            A.shape,
            lambda b, c, i, j: A[b, c, i, j] * Scale[c] + Shift[c],
            name="ScaleShift",
        )

    test_op(compute, input_shapes, out_shape, name=name)


if __name__ == "__main__":
    test_elementwise()
    test_relu()
    test_conv2d_nchw()
    test_depthwise_conv2d_nchw()
    test_pool2d()
    test_softmax()
    test_unary()
    test_is()
    test_bitwise_not()
    test_bitwise_binary()
    test_sigmoid()
    test_matmul()
    test_batch_norm()
