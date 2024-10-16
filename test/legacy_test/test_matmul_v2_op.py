#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from op_test import OpTest, convert_float_to_uint16, get_numeric_gradient
from testsuite import create_op

import paddle
import paddle.distributed as dist
from paddle import base
from paddle.base import core


def reference_matmul(X, Y, transpose_X=False, transpose_Y=False):
    """Reference forward implementation using np.matmul."""
    # np.matmul does not support the transpose flags, so we manually
    # transpose X and Y appropriately.
    if transpose_X:
        if X.ndim == 1:
            X = X.reshape((X.size,))
        elif X.ndim == 2:
            X = X.T
        else:
            dim = list(range(len(X.shape)))
            dim[-1], dim[len(X.shape) - 2] = dim[len(X.shape) - 2], dim[-1]
            X = np.transpose(X, tuple(dim))
    if transpose_Y:
        if Y.ndim == 1:
            Y = Y.reshape((Y.size,))
        else:
            dim = list(range(len(Y.shape)))
            dim[-1], dim[len(Y.shape) - 2] = dim[len(Y.shape) - 2], dim[-1]
            Y = np.transpose(Y, tuple(dim))

    Out = np.matmul(X, Y)
    return Out


class TestMatMulV2Op(OpTest):
    """
    case 1
    """

    def config(self):
        self.x_shape = (100,)
        self.y_shape = (100,)
        self.trans_x = False
        self.trans_y = False

    def init_kernel_type(self):
        self.dtype = "float32" if core.is_compiled_with_rocm() else "float64"

    def setUp(self):
        self.init_kernel_type()
        self.config()
        self.op_type = "matmul_v2"
        self.prim_op_type = "prim"
        self.python_api = paddle.tensor.matmul
        self.public_python_api = paddle.tensor.matmul
        if self.is_bfloat16_op():
            x = np.random.random(self.x_shape).astype(np.float32)
            y = np.random.random(self.y_shape).astype(np.float32)
        else:
            x = np.random.random(self.x_shape).astype(self.dtype)
            y = np.random.random(self.y_shape).astype(self.dtype)
            # -0.1 ~ 0.1
            x = -0.1 + 0.2 * x
            y = -0.1 + 0.2 * y
        result = reference_matmul(x, y, self.trans_x, self.trans_y)
        if self.is_bfloat16_op():
            result = result.astype(np.float32)
            self.inputs = {
                'X': convert_float_to_uint16(x),
                'Y': convert_float_to_uint16(y),
            }
            self.inputs_fp32 = {
                'X': x,
                'Y': y,
            }
        else:
            result = result.astype(self.dtype)
            self.inputs = {
                'X': x,
                'Y': y,
            }
        self.attrs = {'trans_x': self.trans_x, 'trans_y': self.trans_y}
        self.outputs = {'Out': result}

    def test_check_output(self):
        self.check_output(
            check_cinn=self.check_cinn if hasattr(self, 'check_cinn') else True,
            check_pir=True,
        )

    def test_check_grad(self):
        if core.is_compiled_with_rocm():
            self.check_grad(
                ['X', 'Y'],
                'Out',
                max_relative_error=1e-2,
                check_cinn=(
                    self.check_cinn if hasattr(self, 'check_cinn') else True
                ),
                check_pir=True,
                check_prim_pir=True,
            )
        else:
            self.check_grad(
                ['X', 'Y'],
                'Out',
                check_cinn=(
                    self.check_cinn if hasattr(self, 'check_cinn') else True
                ),
                check_pir=True,
                check_prim_pir=True,
            )


class TestMatMulOp2(TestMatMulV2Op):
    """
    case 2
    """

    def config(self):
        self.x_shape = (100,)
        self.y_shape = (1, 3, 2, 100)
        self.trans_x = False
        self.trans_y = True


class TestMatMulOp3(TestMatMulV2Op):
    """
    case 3
    """

    def config(self):
        self.x_shape = (100,)
        self.y_shape = (1, 1, 100, 2)
        self.trans_x = False
        self.trans_y = False
        self.placements = {
            'X': [dist.Shard(0)],
            'Y': [dist.Replicate()],
        }

    def test_check_grad(self):
        if core.is_compiled_with_rocm():
            self.check_grad(
                ['X', 'Y'],
                'Out',
                max_relative_error=1e-2,
                check_cinn=(
                    self.check_cinn if hasattr(self, 'check_cinn') else True
                ),
                check_pir=True,
                check_auto_parallel=True,
                check_prim_pir=True,
            )
        else:
            self.check_grad(
                ['X', 'Y'],
                'Out',
                check_cinn=(
                    self.check_cinn if hasattr(self, 'check_cinn') else True
                ),
                check_pir=True,
                check_auto_parallel=True,
                check_prim_pir=True,
            )


class TestMatMulOp4(TestMatMulV2Op):
    """
    case 4
    """

    def config(self):
        self.x_shape = (100,)
        self.y_shape = (1, 2, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatMulOp5(TestMatMulV2Op):
    """
    case 5
    """

    def config(self):
        self.x_shape = (1, 1, 100, 1)
        self.y_shape = (100,)
        self.trans_x = True
        self.trans_y = False


class TestMatMulOp6(TestMatMulV2Op):
    """
    case 6
    """

    def config(self):
        self.x_shape = (1, 2, 102, 1)
        self.y_shape = (102,)
        self.trans_x = True
        self.trans_y = False


class TestMatMulOp7(TestMatMulV2Op):
    """
    case 7
    """

    def config(self):
        self.x_shape = (1, 2, 1, 100)
        self.y_shape = (100,)
        self.trans_x = False
        self.trans_y = False


class TestMatMulOp8(TestMatMulV2Op):
    """
    case 8
    """

    def config(self):
        self.x_shape = (1, 1, 2, 100)
        self.y_shape = (1, 1, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatMulOp9(TestMatMulV2Op):
    """
    case 9
    """

    def config(self):
        self.x_shape = (1, 1, 1, 100)
        self.y_shape = (2, 1, 2, 100)
        self.trans_x = False
        self.trans_y = True


class TestMatMulOp10(TestMatMulV2Op):
    """
    case 10
    """

    def config(self):
        self.x_shape = (1, 1, 25, 4)
        self.y_shape = (1, 2, 4, 25)
        self.trans_x = False
        self.trans_y = False


class TestMatMulOp11(TestMatMulV2Op):
    """
    case 11
    """

    def config(self):
        self.x_shape = (2, 1, 2, 100)
        self.y_shape = (1, 1, 100, 2)
        self.trans_x = False
        self.trans_y = False


class TestMatMulOp12(TestMatMulV2Op):
    """
    case 12
    """

    def config(self):
        self.x_shape = (2, 1, 4, 25)
        self.y_shape = (1, 1, 4, 25)
        self.trans_x = True
        self.trans_y = False


class TestMatMulOp13(TestMatMulV2Op):
    """
    case 13
    """

    def config(self):
        self.x_shape = (2, 2, 10, 10)
        self.y_shape = (2, 2, 10, 10)
        self.trans_x = True
        self.trans_y = False


class TestMatMulOp14(TestMatMulV2Op):
    """
    case 14_1
    """

    def config(self):
        self.x_shape = (3, 1, 6, 6)
        self.y_shape = (1, 2, 6, 9)
        self.trans_x = True
        self.trans_y = False


class TestMatMulOp15(TestMatMulV2Op):
    """
    case 14_2
    """

    def config(self):
        self.x_shape = (3, 1, 6, 6)
        self.y_shape = (1, 2, 6, 9)
        self.trans_x = False
        self.trans_y = False


class TestMatMulOp16(TestMatMulV2Op):
    """
    case 16 : to check the gradient for special case
    """

    def config(self):
        self.x_shape = 100
        self.y_shape = (1, 2, 2, 100, 2)
        self.trans_x = False
        self.trans_y = False
        self.check_cinn = False


class TestMatMulOp17(TestMatMulV2Op):
    """
    case 17 : to check the gradient for special case
    """

    def config(self):
        self.x_shape = (2, 1, 100)
        self.y_shape = 100
        self.trans_x = False
        self.trans_y = False


class TestMatMulOpBroadcast1(TestMatMulV2Op):
    """
    case 14_3
    """

    def config(self):
        self.x_shape = (3, 1, 10, 10)
        self.y_shape = (1, 2, 10, 10)
        self.trans_x = True
        self.trans_y = True


class TestMatMulOpBroadcast2(TestMatMulV2Op):
    """
    case 14_4
    """

    def config(self):
        self.x_shape = (3, 1, 10, 10)
        self.y_shape = (1, 2, 10, 10)
        self.trans_x = False
        self.trans_y = True


class TestMatMulV2OpAutoParallel(OpTest):
    def config(self):
        self.x_shape = (32, 64)
        self.y_shape = (48, 32)
        self.trans_x = True
        self.trans_y = True
        self.placements = {
            'X': [dist.Shard(1)],
            'Y': [dist.Replicate()],
        }

    def init_kernel_type(self):
        self.dtype = "float32" if core.is_compiled_with_rocm() else "float64"

    def setUp(self):
        self.init_kernel_type()
        self.config()
        self.op_type = "matmul_v2"
        self.prim_op_type = "prim"
        self.python_api = paddle.tensor.matmul
        self.public_python_api = paddle.tensor.matmul
        x = np.random.random(self.x_shape).astype(self.dtype)
        y = np.random.random(self.y_shape).astype(self.dtype)
        # -0.1 ~ 0.1
        x = -0.1 + 0.2 * x
        y = -0.1 + 0.2 * y
        result = reference_matmul(x, y, self.trans_x, self.trans_y)
        self.inputs = {
            'X': x,
            'Y': y,
        }
        self.attrs = {'trans_x': self.trans_x, 'trans_y': self.trans_y}
        self.outputs = {'Out': result}

    def test_check_grad(self):
        if core.is_compiled_with_rocm():
            self.check_grad(
                ['X', 'Y'],
                'Out',
                max_relative_error=1e-2,
                check_auto_parallel=True,
            )
        else:
            self.check_grad(
                ['X', 'Y'],
                'Out',
                check_auto_parallel=True,
                check_prim_pir=True,
            )


# --------------------test matmul fp16--------------------


def create_test_fp16_class(parent, atol=0.001, max_relative_error=1.0):
    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestMatMulOpFp16Case(parent):
        def init_kernel_type(self):
            self.dtype = np.float16

        def test_check_output(self):
            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(0)
                if core.is_float16_supported(place):
                    self.check_output_with_place(
                        place,
                        atol=atol,
                        check_cinn=(
                            self.check_cinn
                            if hasattr(self, 'check_cinn')
                            else True
                        ),
                        check_pir=True,
                    )

        def test_check_grad(self):
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_grad_with_place(
                    place,
                    ['X', 'Y'],
                    'Out',
                    max_relative_error=max_relative_error,
                    check_cinn=(
                        self.check_cinn if hasattr(self, 'check_cinn') else True
                    ),
                    check_pir=True,
                    check_prim_pir=True,
                )

    cls_name = "{}_{}".format(parent.__name__, "Fp16")
    TestMatMulOpFp16Case.__name__ = cls_name
    globals()[cls_name] = TestMatMulOpFp16Case


create_test_fp16_class(TestMatMulV2Op)
create_test_fp16_class(TestMatMulOp2)
create_test_fp16_class(TestMatMulOp3)
create_test_fp16_class(TestMatMulOp4)
create_test_fp16_class(TestMatMulOp5)
create_test_fp16_class(TestMatMulOp6)
create_test_fp16_class(TestMatMulOp7)
create_test_fp16_class(TestMatMulOp8)
create_test_fp16_class(TestMatMulOp9)
create_test_fp16_class(TestMatMulOp10)
create_test_fp16_class(TestMatMulOp11)
create_test_fp16_class(TestMatMulOp12)
create_test_fp16_class(TestMatMulOp13)
create_test_fp16_class(TestMatMulOp14)
create_test_fp16_class(TestMatMulOp15)
create_test_fp16_class(TestMatMulOp16)
create_test_fp16_class(TestMatMulOp17)
create_test_fp16_class(TestMatMulOpBroadcast1)
create_test_fp16_class(TestMatMulOpBroadcast2)

# --------------------test matmul bf16--------------------


def create_test_bf16_class(parent, atol=0.01):
    @unittest.skipIf(
        not core.is_compiled_with_cuda()
        or paddle.is_compiled_with_rocm()
        or not core.is_bfloat16_supported(core.CUDAPlace(0)),
        "core is not compiled with CUDA and not support the bfloat16",
    )
    class TestMatMulOpBf16Case(parent):
        def get_numeric_grad(self, place, check_name):
            scope = core.Scope()
            self._check_grad_helper()
            op = create_op(
                scope, self.op_type, self.inputs, self.outputs, self.attrs
            )
            return get_numeric_gradient(
                place, scope, op, self.inputs_fp32, check_name, ['Out']
            )

        def init_kernel_type(self):
            self.dtype = np.uint16

        def test_check_output(self):
            place = core.CUDAPlace(0)
            self.check_output_with_place(
                place,
                atol=atol,
                check_cinn=(
                    self.check_cinn if hasattr(self, 'check_cinn') else True
                ),
                check_pir=True,
            )

        def test_check_grad_x(self):
            place = core.CUDAPlace(0)
            numeric_grads = self.get_numeric_grad(place, 'X')
            self.check_grad_with_place(
                place,
                ['X'],
                'Out',
                no_grad_set={'Y'},
                max_relative_error=3e-2,
                atol=3e-2,
                user_defined_grads=[numeric_grads],
                check_cinn=(
                    self.check_cinn if hasattr(self, 'check_cinn') else True
                ),
                check_pir=True,
                check_prim_pir=True,
            )

        def test_check_grad_y(self):
            place = core.CUDAPlace(0)
            numeric_grads = self.get_numeric_grad(place, 'Y')
            self.check_grad_with_place(
                place,
                ['Y'],
                'Out',
                no_grad_set={'X'},
                max_relative_error=3e-2,
                atol=3e-2,
                user_defined_grads=[numeric_grads],
                check_cinn=(
                    self.check_cinn if hasattr(self, 'check_cinn') else True
                ),
                check_pir=True,
                check_prim_pir=True,
            )

        def test_check_grad(self):
            pass

    cls_name = "{}_{}".format(parent.__name__, "Bf16")
    TestMatMulOpBf16Case.__name__ = cls_name
    globals()[cls_name] = TestMatMulOpBf16Case


create_test_bf16_class(TestMatMulV2Op)
create_test_bf16_class(TestMatMulOp2)
create_test_bf16_class(TestMatMulOp3)
create_test_bf16_class(TestMatMulOp4)
create_test_bf16_class(TestMatMulOp5)
create_test_bf16_class(TestMatMulOp6)
create_test_bf16_class(TestMatMulOp7)
create_test_bf16_class(TestMatMulOp8)
create_test_bf16_class(TestMatMulOp9)
create_test_bf16_class(TestMatMulOp10)
create_test_bf16_class(TestMatMulOp11)
create_test_bf16_class(TestMatMulOp12)
create_test_bf16_class(TestMatMulOp13)
create_test_bf16_class(TestMatMulOp14)
create_test_bf16_class(TestMatMulOp15)
create_test_bf16_class(TestMatMulOp16)
create_test_bf16_class(TestMatMulOp17)


class TestMatMulV2API(unittest.TestCase):
    def setUp(self):
        self.places = []
        if (
            os.environ.get('FLAGS_CI_both_cpu_and_gpu', 'False').lower()
            in ['1', 'true', 'on']
            or not core.is_compiled_with_cuda()
        ):
            self.places.append(base.CPUPlace())
        if core.is_compiled_with_cuda():
            self.places.append(base.CUDAPlace(0))

    def check_static_result(self, place):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            input_x = paddle.static.data(
                name="input_x", shape=[4, 3], dtype="float32"
            )
            input_y = paddle.static.data(
                name="input_y", shape=[3, 4], dtype="float32"
            )

            result = paddle.matmul(input_x, input_y)

            x_np = np.random.random([4, 3]).astype("float32")
            y_np = np.random.random([3, 4]).astype("float32")

            exe = base.Executor(place)
            fetches = exe.run(
                paddle.static.default_main_program(),
                feed={"input_x": x_np, "input_y": y_np},
                fetch_list=[result],
            )
        paddle.disable_static()

    def test_static(self):
        for place in self.places:
            self.check_static_result(place=place)

    def test_dygraph(self):
        for place in self.places:
            with base.dygraph.guard(place):
                input_x = np.random.random([4, 3]).astype("float64")
                input_y = np.random.random([3, 4]).astype("float64")
                x = paddle.to_tensor(input_x)
                y = paddle.to_tensor(input_y)
                result = paddle.matmul(x, y)

    def test_dygraph_fp16(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                with base.dygraph.guard(place):
                    input_x = np.random.random([4, 3]).astype("float16")
                    input_y = np.random.random([3, 4]).astype("float16")
                    x = paddle.to_tensor(input_x)
                    y = paddle.to_tensor(input_y)
                    result = paddle.matmul(x, y)

    def test_compute_type_fp32(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                with base.dygraph.guard(place):
                    paddle.set_flags(
                        {'FLAGS_gemm_use_half_precision_compute_type': False}
                    )
                    input_x = np.random.random([2, 8, 16]).astype("float16")
                    input_y = np.random.random([2, 16, 8]).astype("float16")
                    for i in range(0, 16, 2):
                        input_x[:, :, i] += 60000
                        input_x[:, :, i + 1] -= 60000
                    input_y[:, :, :] = 1.5

                    x = paddle.to_tensor(input_x)
                    y = paddle.to_tensor(input_y)
                    result = paddle.matmul(x, y)
                    result_np = np.matmul(input_x, input_y)
                    self.assertTrue(paddle.isfinite(result)[0, 0, 0])
                    self.assertTrue(np.isfinite(result_np)[0, 0, 0])
                    np.testing.assert_array_equal(result_np, result.numpy())
                    paddle.set_flags(
                        {'FLAGS_gemm_use_half_precision_compute_type': True}
                    )

    def test_compute_type_fp16_nan(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                with base.dygraph.guard(place):
                    paddle.set_flags(
                        {'FLAGS_gemm_use_half_precision_compute_type': True}
                    )
                    input_x = np.random.random([2, 8, 16]).astype("float16")
                    input_y = np.random.random([2, 16, 8]).astype("float16")
                    for i in range(0, 16, 2):
                        input_x[:, :, i] += 60000
                        input_x[:, :, i + 1] -= 60000
                    input_y[:, :, :] = 1.5

                    x = paddle.to_tensor(input_x)
                    y = paddle.to_tensor(input_y)
                    result = paddle.matmul(x, y)
                    result_np = np.matmul(input_x, input_y)
                    self.assertFalse(
                        paddle.isfinite(result)[0, 0, 0]
                    )  # contains nan/inf
                    self.assertTrue(np.isfinite(result_np)[0, 0, 0])
                    paddle.set_flags(
                        {'FLAGS_gemm_use_half_precision_compute_type': False}
                    )


class TestComplexMatMulOp(OpTest):
    def setUp(self):
        self.op_type = "matmul_v2"
        self.python_api = paddle.tensor.matmul
        self.init_base_dtype()
        self.init_input_output()

        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(self.x),
            'Y': OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.attrs = {'axis': -1, 'use_mkldnn': False}
        self.outputs = {'Out': self.out}

    def init_base_dtype(self):
        self.dtype = np.complex128

    def init_input_output(self):
        self.x = np.random.random((10, 10)).astype(
            self.dtype
        ) + 1j * np.random.random((10, 10)).astype(self.dtype)
        self.y = np.random.random((10, 10)).astype(
            self.dtype
        ) + 1j * np.random.random((10, 10)).astype(self.dtype)
        self.out = np.dot(self.x, self.y)

    def test_check_output(self):
        self.check_output(check_cinn=False)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_cinn=False,
        )

    def test_check_grad_ignore_x(self):
        self.check_grad(
            ['Y'],
            'Out',
            no_grad_set=set("X"),
            check_cinn=False,
        )

    def test_check_grad_ignore_y(self):
        self.check_grad(
            ['X'],
            'Out',
            no_grad_set=set('Y'),
            check_cinn=False,
        )


class TestComplexMatMulOpBroadcast(OpTest):
    def setUp(self):
        self.op_type = "matmul_v2"
        self.python_api = paddle.tensor.matmul
        self.init_base_dtype()
        self.init_input_output()

        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(self.x),
            'Y': OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.attrs = {'axis': -1, 'use_mkldnn': False}
        self.outputs = {'Out': self.out}

    def init_base_dtype(self):
        self.dtype = np.complex128

    def init_input_output(self):
        self.x = np.random.random((10, 2, 5)).astype(
            self.dtype
        ) + 1j * np.random.random((10, 2, 5)).astype(self.dtype)
        self.y = np.random.random((5, 20)).astype(
            self.dtype
        ) + 1j * np.random.random((5, 20)).astype(self.dtype)
        self.out = np.dot(self.x, self.y)

    def test_check_output(self):
        self.check_output(check_cinn=False)

    def test_check_grad_normal(self):
        self.check_grad(
            ['X', 'Y'],
            'Out',
            check_cinn=False,
        )

    def test_check_grad_ignore_x(self):
        self.check_grad(
            ['Y'],
            'Out',
            no_grad_set=set("X"),
            check_cinn=False,
        )

    def test_check_grad_ignore_y(self):
        self.check_grad(
            ['X'],
            'Out',
            no_grad_set=set('Y'),
            check_cinn=False,
        )


class TestMatMulTypePromotion(TestComplexMatMulOp):
    def init_input_output(self):
        self.x = np.random.random((10, 10)).astype(self.dtype)
        self.y = np.random.random((10, 10)).astype(
            self.dtype
        ) + 1j * np.random.random((10, 10)).astype(self.dtype)
        self.out = np.dot(self.x, self.y)


class TestInt32MatmulOp(OpTest):
    def setUp(self):
        self.op_type = "matmul_v2"
        self.python_api = paddle.tensor.matmul
        self.init_base_dtype()
        self.init_input_output()

        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(self.x),
            'Y': OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.attrs = {'axis': -1, 'use_mkldnn': False}
        self.outputs = {'Out': self.out}

    def init_base_dtype(self):
        self.dtype = np.int32

    def init_input_output(self):
        self.x = np.random.random((10, 10)).astype(self.dtype)
        self.y = np.random.random((10, 10)).astype(self.dtype)
        self.out = np.matmul(self.x, self.y)

    def test_check_output(self):
        self.check_output(check_cinn=False, check_pir=True)


class TestInt32MatMulOpBroadcast(OpTest):
    def setUp(self):
        self.op_type = "matmul_v2"
        self.python_api = paddle.tensor.matmul
        self.init_base_dtype()
        self.init_input_output()

        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(self.x),
            'Y': OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.attrs = {'axis': -1, 'use_mkldnn': False}
        self.outputs = {'Out': self.out}

    def init_base_dtype(self):
        self.dtype = np.int32

    def init_input_output(self):
        self.x = np.random.random((10, 2, 5)).astype(self.dtype)
        self.y = np.random.random((5, 20)).astype(self.dtype)
        self.out = np.matmul(self.x, self.y)

    def test_check_output(self):
        self.check_output(check_cinn=False)


class TestInt64MatmulOp(OpTest):
    def setUp(self):
        self.op_type = "matmul_v2"
        self.python_api = paddle.tensor.matmul
        self.init_base_dtype()
        self.init_input_output()

        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(self.x),
            'Y': OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.attrs = {'axis': -1, 'use_mkldnn': False}
        self.outputs = {'Out': self.out}

    def init_base_dtype(self):
        self.dtype = np.int64

    def init_input_output(self):
        self.x = np.random.random((10, 10)).astype(self.dtype)
        self.y = np.random.random((10, 10)).astype(self.dtype)
        self.out = np.matmul(self.x, self.y)

    def test_check_output(self):
        self.check_output(check_cinn=False, check_pir=True)


class TestInt64MatMulOpBroadcast(OpTest):
    def setUp(self):
        self.op_type = "matmul_v2"
        self.python_api = paddle.tensor.matmul
        self.init_base_dtype()
        self.init_input_output()

        self.inputs = {
            'X': OpTest.np_dtype_to_base_dtype(self.x),
            'Y': OpTest.np_dtype_to_base_dtype(self.y),
        }
        self.attrs = {'axis': -1, 'use_mkldnn': False}
        self.outputs = {'Out': self.out}

    def init_base_dtype(self):
        self.dtype = np.int64

    def init_input_output(self):
        self.x = np.random.random((10, 2, 5)).astype(self.dtype)
        self.y = np.random.random((5, 20)).astype(self.dtype)
        self.out = np.matmul(self.x, self.y)

    def test_check_output(self):
        self.check_output(check_cinn=False)


class TestMatmulop(unittest.TestCase):
    def func_dygraph_matmul(self):
        paddle.disable_static()

        np_a = np.random.random((2, 4)).astype(np.float32)
        np_b = np.random.random((4, 2)).astype(np.float32)

        tensor_a = paddle.to_tensor(np_a, dtype="float32")
        tensor_b = paddle.to_tensor(np_b, dtype="float32")

        # normal case: tensor @ nparray
        expect_out = np_a @ np_b
        actual_out = tensor_a @ np_b
        np.testing.assert_allclose(actual_out, expect_out)

        paddle.enable_static()


if __name__ == "__main__":
    paddle.enable_static()
    unittest.main()
