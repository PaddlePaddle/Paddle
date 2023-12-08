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


def adaptive_start_index(index, input_size, output_size):
    return int(np.floor(index * input_size / output_size))


def adaptive_end_index(index, input_size, output_size):
    return int(np.ceil((index + 1) * input_size / output_size))


def fractional_rational_u(u, alpha, input, output):
    base = input // output

    u_max1 = (base + 2) / alpha - 1
    u_max2 = (input + 1 - base) / alpha - (output - 1)
    max_u = min(u_max1, u_max2)

    return u * max_u


def fractional_start_index(idx, alpha, u):
    return int(np.ceil(alpha * (idx + u) - 1))


def fractional_end_index(idx, alpha, u):
    return int(np.ceil(alpha * (idx + 1 + u) - 1))


def fractional_max_pool3D_forward_naive(
    x,
    ksize,
    strides,
    paddings,
    global_pool=False,
    adaptive=False,
    fractional=False,
    random_u=None,
):
    N, C, D, H, W = x.shape
    if global_pool:
        ksize = [D, H, W]
        paddings = [0, 0, 0]

    if adaptive or fractional:
        D_out, H_out, W_out = ksize
    else:
        D_out = (D - ksize[0] + 2 * paddings[0]) // strides[0] + 1
        H_out = (H - ksize[1] + 2 * paddings[1]) // strides[1] + 1
        W_out = (W - ksize[2] + 2 * paddings[2]) // strides[2] + 1

    if fractional:
        u = random_u

        alpha_depth = D / D_out
        alpha_height = H / H_out
        alpha_width = W / W_out

        u_depth = fractional_rational_u(u, alpha_depth, D, D_out)
        u_height = fractional_rational_u(u, alpha_height, H, H_out)
        u_width = fractional_rational_u(u, alpha_width, W, W_out)

    out = np.zeros((N, C, D_out, H_out, W_out))
    mask = np.zeros((N, C, D_out, H_out, W_out))
    for k in range(D_out):
        if adaptive:
            d_start = adaptive_start_index(k, D, ksize[0])
            d_end = adaptive_end_index(k, D, ksize[0])
        elif fractional:
            d_start = fractional_start_index(k, alpha_depth, u_depth)
            d_end = fractional_end_index(k, alpha_depth, u_depth)
            d_start = max(d_start, 0)
            d_end = min(d_end, D)
        else:
            d_start = np.max((k * strides[0] - paddings[0], 0))
            d_end = np.min((k * strides[0] + ksize[0] - paddings[0], D))
        for i in range(H_out):
            if adaptive:
                h_start = adaptive_start_index(i, H, ksize[1])
                h_end = adaptive_end_index(i, H, ksize[1])
            elif fractional:
                h_start = fractional_start_index(i, alpha_height, u_height)
                h_end = fractional_end_index(i, alpha_height, u_height)
                h_start = max(h_start, 0)
                h_end = min(h_end, H)
            else:
                h_start = np.max((i * strides[1] - paddings[1], 0))
                h_end = np.min((i * strides[1] + ksize[1] - paddings[1], H))
            for j in range(W_out):
                if adaptive:
                    w_start = adaptive_start_index(j, W, ksize[2])
                    w_end = adaptive_end_index(j, W, ksize[2])
                elif fractional:
                    w_start = fractional_start_index(j, alpha_width, u_width)
                    w_end = fractional_end_index(j, alpha_width, u_width)
                    w_start = max(w_start, 0)
                    w_end = min(w_end, W)
                else:
                    w_start = np.max((j * strides[2] - paddings[2], 0))
                    w_end = np.min((j * strides[2] + ksize[2] - paddings[2], W))
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


# ----------------fractional_max_pool3d_with_index----------------
def fractional_max_pool3d_with_index_wapper(
    x,
    kernel_size=[],
    strides=[],
    paddings=[],
    global_pooling=False,
    adaptive=False,
    fractional=False,
    random_u=None,
):
    return paddle._C_ops.fractional_max_pool3d_with_index(
        x,
        kernel_size,
        strides,
        paddings,
        global_pooling,
        adaptive,
        fractional,
        random_u,
    )


class TestMaxPoolWithIndex_Op(OpTest):
    def setUp(self):
        self.init_test_case()
        self.init_global()
        self.init_adaptive()
        self.init_fractional()
        self.init_dtype()

        if self.is_bfloat16_op():
            input = np.random.random(self.shape).astype(np.float32)
            input = convert_uint16_to_float(
                convert_float_to_uint16(np.round(input * 100.0, 2))
            )

        else:
            input = np.random.random(self.shape).astype(self.dtype)
            input = np.round(input * 100.0, 2)

        output, mask = self.pool_forward_naive(
            input,
            self.ksize,
            self.strides,
            self.paddings,
            self.global_pool,
            self.adaptive,
            self.fractional,
            self.random_u,
        )
        mask = mask.astype("int32")
        if self.is_bfloat16_op():
            output = output.astype(np.float32)
        else:
            output = output.astype(self.dtype)

        self.attrs = {
            'strides': self.strides,
            'paddings': self.paddings,
            'ksize': self.ksize,
            'global_pooling': self.global_pool,
            'adaptive': self.adaptive,
            'fractional': self.fractional,
            'random_u': self.random_u,
        }

        if self.is_bfloat16_op():
            self.inputs = {'X': convert_float_to_uint16(input)}
            self.outputs = {
                'Out': convert_float_to_uint16(output),
                "Mask": mask,
            }
            self.inputs_fp32 = {'X': input}

        else:
            self.inputs = {'X': input}
            self.outputs = {'Out': output, "Mask": mask}

    def init_dtype(self):
        self.dtype = np.float64

    def test_check_output(self):
        self.check_output()

    def test_check_grad(self):
        self.check_grad({'X'}, ['Out'])

    def init_test_case(self):
        self.op_type = "fractional_max_pool3d_with_index"
        self.python_api = fractional_max_pool3d_with_index_wapper
        self.pool_forward_naive = fractional_max_pool3D_forward_naive
        self.shape = [2, 3, 7, 7, 7]
        self.ksize = [3, 3, 3]
        self.strides = [2, 2, 2]
        self.paddings = [1, 1, 1]

    def init_global(self):
        self.global_pool = False

    def init_adaptive(self):
        self.adaptive = False

    def init_fractional(self):
        self.fractional = False
        self.random_u = 0.3


class TestCase1(TestMaxPoolWithIndex_Op):
    def init_global(self):
        self.global_pool = True


class TestCase2(TestMaxPoolWithIndex_Op):
    def init_test_case(self):
        self.op_type = "fractional_max_pool3d_with_index"
        self.python_api = fractional_max_pool3d_with_index_wapper
        self.pool_forward_naive = fractional_max_pool3D_forward_naive
        self.shape = [2, 3, 7, 7, 7]
        self.ksize = [3, 3, 3]
        self.strides = [2, 2, 2]
        self.paddings = [0, 0, 0]

    def init_global(self):
        self.global_pool = True


class TestCase3(TestCase2):
    def init_global(self):
        self.global_pool = False


class TestCastAdaptive3d(TestMaxPoolWithIndex_Op):
    def init_adaptive(self):
        self.adaptive = True


class TestCastFractional3d(TestMaxPoolWithIndex_Op):
    def init_fractional(self):
        self.fractional = True
        self.random_u = 0.3


# ----------------fractional_max_pool3d_with_index_fp16----------------
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
                self.check_grad_with_place(place, {'X'}, ['Out'])

    cls_name = "{}_{}".format(parent.__name__, "FP16OP")
    TestMaxPool3dFP16.__name__ = cls_name
    globals()[cls_name] = TestMaxPool3dFP16


create_test_fp16_class(TestMaxPoolWithIndex_Op)
create_test_fp16_class(TestCase1)
create_test_fp16_class(TestCase2)
create_test_fp16_class(TestCase3)
create_test_fp16_class(TestCastAdaptive3d)
create_test_fp16_class(TestCastFractional3d)


# ----------------fractional_max_pool3d_with_index_bf16----------------
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
                place, scope, op, self.inputs_fp32, check_name, ['Out']
            )

        def test_check_output(self):
            place = core.CUDAPlace(0)
            if core.is_bfloat16_supported(place):
                self.check_output_with_place(place)

        def test_check_grad(self):
            place = core.CUDAPlace(0)
            numeric_grads = self.get_numeric_grad(place, 'X')
            if core.is_bfloat16_supported(place):
                self.check_grad_with_place(
                    place,
                    {'X'},
                    ['Out'],
                )

    cls_name = "{}_{}".format(parent.__name__, "BF16OP")
    TestMaxPool3dBF16.__name__ = cls_name
    globals()[cls_name] = TestMaxPool3dBF16


create_test_bf16_class(TestMaxPoolWithIndex_Op)
create_test_bf16_class(TestCase1)
create_test_bf16_class(TestCase2)
create_test_bf16_class(TestCase3)
create_test_bf16_class(TestCastAdaptive3d)
create_test_bf16_class(TestCastFractional3d)


if __name__ == '__main__':
    unittest.main()
