#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
from op_test import (
    OpTest,
    convert_float_to_uint16,
    convert_uint16_to_float,
    get_numeric_gradient,
)
from testsuite import create_op

import paddle
from paddle.base import core


def fractional_rational_u(u, alpha, input, output, pool_size=0):
    if pool_size > 0:
        return u

    base = input // output

    u_max1 = (base + 2) / alpha - 1
    u_max2 = (input + 1 - base) / alpha - (output - 1)
    max_u = min(u_max1, u_max2)

    return u * max_u


def fractional_start_index(idx, alpha, u, pool_size=0):
    return int((idx + u) * alpha) - int(u * alpha)


def fractional_end_index(idx, alpha, u, pool_size=0):
    if pool_size > 0:
        return int((idx + u) * alpha) - int(u * alpha) + pool_size
    return int((idx + 1 + u) * alpha) - int(u * alpha)


def fractional_max_pool3D_forward_naive(
    x,
    output_size,
    kernel_size=[0, 0, 0],
    random_u=None,
    return_mask=True,
):
    N, C, D, H, W = x.shape
    D_out, H_out, W_out = output_size
    pool_depth, pool_height, pool_width = kernel_size

    u = random_u

    alpha_depth = (D - pool_depth) / (D_out - (1 if pool_depth > 0 else 0))
    alpha_height = (H - pool_height) / (H_out - (1 if pool_height > 0 else 0))
    alpha_width = (W - pool_width) / (W_out - (1 if pool_width > 0 else 0))

    u_depth = fractional_rational_u(u, alpha_depth, D, D_out, pool_depth)
    u_height = fractional_rational_u(u, alpha_height, H, H_out, pool_height)
    u_width = fractional_rational_u(u, alpha_width, W, W_out, pool_width)

    out = np.zeros((N, C, D_out, H_out, W_out))
    mask = np.zeros((N, C, D_out, H_out, W_out))
    for k in range(D_out):
        d_start = fractional_start_index(k, alpha_depth, u_depth, pool_depth)
        d_end = fractional_end_index(k, alpha_depth, u_depth, pool_depth)
        d_start = max(d_start, 0)
        d_end = min(d_end, D)

        for i in range(H_out):
            h_start = fractional_start_index(
                i, alpha_height, u_height, pool_height
            )
            h_end = fractional_end_index(i, alpha_height, u_height, pool_height)
            h_start = max(h_start, 0)
            h_end = min(h_end, H)

            for j in range(W_out):
                w_start = fractional_start_index(
                    j, alpha_width, u_width, pool_width
                )
                w_end = fractional_end_index(
                    j, alpha_width, u_width, pool_width
                )
                w_start = max(w_start, 0)
                w_end = min(w_end, W)

                x_masked = x[:, :, d_start:d_end, h_start:h_end, w_start:w_end]

                out[:, :, k, i, j] = np.max(x_masked, axis=(2, 3, 4))

                for n in range(N):
                    for c in range(C):
                        arr = x_masked[n, c, :, :, :]
                        index = np.where(arr == np.max(arr))
                        sub_deep = index[0][0]
                        sub_row = index[1][0]
                        sub_col = index[2][0]
                        index = (
                            ((d_start + sub_deep) * H + (h_start + sub_row)) * W
                            + w_start
                            + sub_col
                        )
                        mask[n, c, k, i, j] = index

    return out, mask


# ----------------fractional_max_pool3d----------------
def fractional_max_pool3d_wrapper(
    x,
    output_size=None,
    kernel_size=[0, 0, 0],
    random_u=None,
    return_mask=True,
):
    return paddle._C_ops.fractional_max_pool3d(
        x,
        output_size,
        kernel_size,
        random_u,
        return_mask,
    )


class TestMaxPoolWithIndex_Op(OpTest):
    def setUp(self):
        self.op_type = "fractional_max_pool3d"
        self.python_api = fractional_max_pool3d_wrapper
        self.pool_forward_naive = fractional_max_pool3D_forward_naive

        self.init_test_case()
        self.init_fractional()
        self.init_dtype()

        if self.is_bfloat16_op():
            np.random.seed(2023)
            input = np.random.random(self.shape).astype(np.float32)
            input = convert_uint16_to_float(
                convert_float_to_uint16(np.round(input * 100.0, 2))
            )

        else:
            np.random.seed(2023)
            input = np.random.random(self.shape).astype(self.dtype)
            input = np.round(input * 100.0, 2)

        output, mask = self.pool_forward_naive(
            input,
            self.output_size,
            self.kernel_size,
            self.random_u,
            self.return_mask,
        )
        mask = mask.astype("int32")
        if self.is_bfloat16_op():
            output = output.astype(np.float32)
        else:
            output = output.astype(self.dtype)

        self.attrs = {
            'output_size': self.output_size,
            'kernel_size': self.kernel_size,
            'random_u': self.random_u,
            'return_mask': self.return_mask,
        }

        if self.is_bfloat16_op():
            self.inputs = {'x': convert_float_to_uint16(input)}
            self.outputs = {
                'out': convert_float_to_uint16(output),
                'mask': mask,
            }
            self.inputs_fp32 = {'x': input}

        else:
            self.inputs = {'x': input}
            self.outputs = {'out': output, 'mask': mask}

    def init_dtype(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad({'x'}, ['out'])

    def init_test_case(self):
        self.shape = [2, 3, 7, 7, 7]
        self.output_size = [3, 3, 3]
        self.kernel_size = [0, 0, 0]
        self.return_mask = True

    def init_fractional(self):
        self.random_u = 0.3


class TestCase1(TestMaxPoolWithIndex_Op):
    def init_test_case(self):
        self.shape = [2, 5, 9, 9, 9]
        self.output_size = [5, 5, 5]
        self.kernel_size = [0, 0, 0]
        self.return_mask = False


class TestCase2(TestCase1):
    def init_fractional(self):
        self.random_u = 0.5


class TestCase3(TestMaxPoolWithIndex_Op):
    def init_test_case(self):
        self.shape = [2, 5, 7, 7, 7]
        self.output_size = [5, 5, 5]
        self.kernel_size = [2, 2, 2]
        self.return_mask = True


# ----------------fractional_max_pool3d_fp16----------------
def create_test_fp16_class(parent):
    @unittest.skipIf(
        not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
    )
    class TestMaxPool3dFP16(parent):
        def init_dtype(self):
            self.dtype = np.float16

        def test_check_output(self):
            if core.is_compiled_with_cuda():
                place = core.CUDAPlace(0)
                if core.is_float16_supported(place):
                    self.check_output_with_place(place)

        def test_check_grad(self):
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_grad_with_place(place, {'x'}, ['out'])

    cls_name = "{}_{}".format(parent.__name__, "FP16OP")
    TestMaxPool3dFP16.__name__ = cls_name
    globals()[cls_name] = TestMaxPool3dFP16


create_test_fp16_class(TestMaxPoolWithIndex_Op)
create_test_fp16_class(TestCase1)
create_test_fp16_class(TestCase2)
create_test_fp16_class(TestCase3)


# ----------------fractional_max_pool3d_bf16----------------
def create_test_bf16_class(parent):
    @unittest.skipIf(
        not core.is_compiled_with_cuda()
        or not core.is_bfloat16_supported(core.CUDAPlace(0)),
        "core is not compiled with CUDA and do not support bfloat16",
    )
    class TestMaxPool3dBF16(parent):
        def init_dtype(self):
            self.dtype = np.uint16

        def get_numeric_grad(self, place, check_name):
            scope = core.Scope()
            self._check_grad_helper()
            op = create_op(
                scope, self.op_type, self.inputs, self.outputs, self.attrs
            )
            return get_numeric_gradient(
                place, scope, op, self.inputs_fp32, check_name, ['out']
            )

        def test_check_output(self):
            place = core.CUDAPlace(0)
            if core.is_bfloat16_supported(place):
                self.check_output_with_place(place)

        def test_check_grad(self):
            place = core.CUDAPlace(0)
            numeric_grads = self.get_numeric_grad(place, 'x')
            if core.is_bfloat16_supported(place):
                self.check_grad_with_place(
                    place, {'x'}, ['out'], user_defined_grads=[numeric_grads]
                )

    cls_name = "{}_{}".format(parent.__name__, "BF16OP")
    TestMaxPool3dBF16.__name__ = cls_name
    globals()[cls_name] = TestMaxPool3dBF16


create_test_bf16_class(TestMaxPoolWithIndex_Op)
create_test_bf16_class(TestCase1)
create_test_bf16_class(TestCase2)
create_test_bf16_class(TestCase3)


if __name__ == '__main__':
    unittest.main()
