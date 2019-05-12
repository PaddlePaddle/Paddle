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
from __future__ import division

import unittest
import numpy as np

import paddle.fluid.core as core
from op_test import OpTest


def adaptive_start_index(index, input_size, output_size):
    return int(np.floor(index * input_size / output_size))


def adaptive_end_index(index, input_size, output_size):
    return int(np.ceil((index + 1) * input_size / output_size))


def max_pool3D_forward_naive(x,
                             ksize,
                             strides,
                             paddings,
                             global_pool=0,
                             ceil_mode=False,
                             exclusive=True,
                             adaptive=False):
    N, C, D, H, W = x.shape
    if global_pool == 1:
        ksize = [D, H, W]
    if adaptive:
        D_out, H_out, W_out = ksize
    else:
        D_out = (D - ksize[0] + 2 * paddings[0] + strides[0] - 1
                 ) // strides[0] + 1 if ceil_mode else (
                     H - ksize[0] + 2 * paddings[0]) // strides[0] + 1
        H_out = (H - ksize[1] + 2 * paddings[1] + strides[1] - 1
                 ) // strides[1] + 1 if ceil_mode else (
                     W - ksize[1] + 2 * paddings[1]) // strides[1] + 1
        W_out = (W - ksize[2] + 2 * paddings[2] + strides[2] - 1
                 ) // strides[2] + 1 if ceil_mode else (
                     W - ksize[2] + 2 * paddings[2]) // strides[2] + 1
    out = np.zeros((N, C, D_out, H_out, W_out))
    for k in range(D_out):
        if adaptive:
            d_start = adaptive_start_index(k, D, ksize[0])
            d_end = adaptive_end_index(k, D, ksize[0])
        else:
            d_start = np.max((k * strides[0] - paddings[0], 0))
            d_end = np.min((k * strides[0] + ksize[0] - paddings[0], D))
        for i in range(H_out):
            if adaptive:
                h_start = adaptive_start_index(i, H, ksize[1])
                h_end = adaptive_end_index(i, H, ksize[1])
            else:
                h_start = np.max((i * strides[1] - paddings[1], 0))
                h_end = np.min((i * strides[1] + ksize[1] - paddings[1], H))
            for j in range(W_out):
                if adaptive:
                    w_start = adaptive_start_index(j, W, ksize[2])
                    w_end = adaptive_end_index(j, W, ksize[2])
                else:
                    w_start = np.max((j * strides[2] - paddings[2], 0))
                    w_end = np.min((j * strides[2] + ksize[2] - paddings[2], W))
                x_masked = x[:, :, d_start:d_end, h_start:h_end, w_start:w_end]

                out[:, :, k, i, j] = np.max(x_masked, axis=(2, 3, 4))
    return out


def avg_pool3D_forward_naive(x,
                             ksize,
                             strides,
                             paddings,
                             global_pool=0,
                             ceil_mode=False,
                             exclusive=True,
                             adaptive=False):
    N, C, D, H, W = x.shape
    if global_pool == 1:
        ksize = [D, H, W]
    if adaptive:
        D_out, H_out, W_out = ksize
    else:
        D_out = (D - ksize[0] + 2 * paddings[0] + strides[0] - 1
                 ) // strides[0] + 1 if ceil_mode else (
                     H - ksize[0] + 2 * paddings[0]) // strides[0] + 1
        H_out = (H - ksize[1] + 2 * paddings[1] + strides[1] - 1
                 ) // strides[1] + 1 if ceil_mode else (
                     W - ksize[1] + 2 * paddings[1]) // strides[1] + 1
        W_out = (W - ksize[2] + 2 * paddings[2] + strides[2] - 1
                 ) // strides[2] + 1 if ceil_mode else (
                     W - ksize[2] + 2 * paddings[2]) // strides[2] + 1
    out = np.zeros((N, C, D_out, H_out, W_out))
    for k in range(D_out):
        if adaptive:
            d_start = adaptive_start_index(k, D, ksize[0])
            d_end = adaptive_end_index(k, D, ksize[0])
        else:
            d_start = np.max((k * strides[0] - paddings[0], 0))
            d_end = np.min((k * strides[0] + ksize[0] - paddings[0], D))
        for i in range(H_out):
            if adaptive:
                h_start = adaptive_start_index(i, H, ksize[1])
                h_end = adaptive_end_index(i, H, ksize[1])
            else:
                h_start = np.max((i * strides[1] - paddings[1], 0))
                h_end = np.min((i * strides[1] + ksize[1] - paddings[1], H))
            for j in range(W_out):
                if adaptive:
                    w_start = adaptive_start_index(j, W, ksize[2])
                    w_end = adaptive_end_index(j, W, ksize[2])
                else:
                    w_start = np.max((j * strides[2] - paddings[2], 0))
                    w_end = np.min((j * strides[2] + ksize[2] - paddings[2], W))
                x_masked = x[:, :, d_start:d_end, h_start:h_end, w_start:w_end]

                field_size = (d_end - d_start) * (h_end - h_start) * (w_end - w_start) \
                             if (exclusive or adaptive) else ksize[0] * ksize[1] * ksize[2]
                out[:, :, k, i, j] = np.sum(x_masked, axis=(2, 3,
                                                            4)) / field_size
    return out


class TestPool3d_Op(OpTest):
    def setUp(self):
        self.op_type = "pool3d"
        self.use_cudnn = False
        self.dtype = np.float32
        self.init_test_case()
        self.init_global_pool()
        self.init_kernel_type()
        self.init_pool_type()
        self.init_ceil_mode()
        self.init_exclusive()
        self.init_adaptive()

        if self.global_pool:
            self.paddings = [0 for _ in range(len(self.paddings))]
        input = np.random.random(self.shape).astype(self.dtype)
        output = self.pool3D_forward_naive(
            input, self.ksize, self.strides, self.paddings, self.global_pool,
            self.ceil_mode, self.exclusive, self.adaptive).astype(self.dtype)
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(input)}

        self.attrs = {
            'strides': self.strides,
            'paddings': self.paddings,
            'ksize': self.ksize,
            'pooling_type': self.pool_type,
            'global_pooling': self.global_pool,
            'use_cudnn': self.use_cudnn,
            'ceil_mode': self.ceil_mode,
            'data_format':
            'AnyLayout',  # TODO(dzhwinter) : should be fix latter
            'exclusive': self.exclusive,
            'adaptive': self.adaptive
        }

        self.outputs = {'Out': output}

    def has_cudnn(self):
        return core.is_compiled_with_cuda() and self.use_cudnn

    def test_check_output(self):
        if self.has_cudnn():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-5)
        else:
            self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        if self.has_cudnn() and self.pool_type != "max":
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place, set(['X']), 'Out', max_relative_error=0.07)
        elif self.pool_type != "max":
            self.check_grad(set(['X']), 'Out', max_relative_error=0.07)

    def init_test_case(self):
        self.shape = [2, 3, 5, 5, 5]
        self.ksize = [3, 3, 3]
        self.strides = [1, 1, 1]
        self.paddings = [0, 0, 0]

    def init_kernel_type(self):
        pass

    def init_pool_type(self):
        self.pool_type = "avg"
        self.pool3D_forward_naive = avg_pool3D_forward_naive

    def init_global_pool(self):
        self.global_pool = True

    def init_ceil_mode(self):
        self.ceil_mode = False

    def init_exclusive(self):
        self.exclusive = True

    def init_adaptive(self):
        self.adaptive = False


class TestCase1(TestPool3d_Op):
    def init_test_case(self):
        self.shape = [2, 3, 7, 7, 7]
        self.ksize = [3, 3, 3]
        self.strides = [1, 1, 1]
        self.paddings = [0, 0, 0]

    def init_pool_type(self):
        self.pool_type = "avg"
        self.pool3D_forward_naive = avg_pool3D_forward_naive

    def init_global_pool(self):
        self.global_pool = False


class TestCase2(TestPool3d_Op):
    def init_test_case(self):
        self.shape = [2, 3, 7, 7, 7]
        self.ksize = [3, 3, 3]
        self.strides = [1, 1, 1]
        self.paddings = [1, 1, 1]

    def init_pool_type(self):
        self.pool_type = "avg"
        self.pool3D_forward_naive = avg_pool3D_forward_naive

    def init_global_pool(self):
        self.global_pool = False


class TestCase3(TestPool3d_Op):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool3D_forward_naive = max_pool3D_forward_naive


class TestCase4(TestCase1):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool3D_forward_naive = max_pool3D_forward_naive


class TestCase5(TestCase2):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool3D_forward_naive = max_pool3D_forward_naive


#--------------------test pool3d--------------------
class TestCUDNNCase1(TestPool3d_Op):
    def init_kernel_type(self):
        self.use_cudnn = True


class TestFP16CUDNNCase1(TestPool3d_Op):
    def init_kernel_type(self):
        self.use_cudnn = True
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestCUDNNCase2(TestCase1):
    def init_kernel_type(self):
        self.use_cudnn = True


class TestFP16CUDNNCase2(TestCase1):
    def init_kernel_type(self):
        self.use_cudnn = True
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestCUDNNCase3(TestCase2):
    def init_kernel_type(self):
        self.use_cudnn = True


class TestFP16CUDNNCase3(TestCase2):
    def init_kernel_type(self):
        self.use_cudnn = True
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestCUDNNCase4(TestCase3):
    def init_kernel_type(self):
        self.use_cudnn = True


class TestFP16CUDNNCase4(TestCase3):
    def init_kernel_type(self):
        self.use_cudnn = True
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestCUDNNCase5(TestCase4):
    def init_kernel_type(self):
        self.use_cudnn = True


class TestFP16CUDNNCase5(TestCase4):
    def init_kernel_type(self):
        self.use_cudnn = True
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestCUDNNCase6(TestCase5):
    def init_kernel_type(self):
        self.use_cudnn = True


class TestFP16CUDNNCase6(TestCase5):
    def init_kernel_type(self):
        self.use_cudnn = True
        self.dtype = np.float16

    def test_check_output(self):
        if core.is_compiled_with_cuda():
            place = core.CUDAPlace(0)
            if core.is_float16_supported(place):
                self.check_output_with_place(place, atol=1e-3)


class TestCeilModeCase1(TestCUDNNCase1):
    def init_ceil_mode(self):
        self.ceil_mode = True


class TestCeilModeCase2(TestCUDNNCase2):
    def init_ceil_mode(self):
        self.ceil_mode = True


class TestCeilModeCase3(TestCase1):
    def init_ceil_mode(self):
        self.ceil_mode = True


class TestCeilModeCase4(TestCase2):
    def init_ceil_mode(self):
        self.ceil_mode = True


class TestAvgInclude(TestCase2):
    def init_exclusive(self):
        self.exclusive = False


class TestCUDNNAvgInclude(TestCUDNNCase3):
    def init_exclusive(self):
        self.exclusive = False


class TestAvgPoolAdaptive(TestCase1):
    def init_adaptive(self):
        self.adaptive = True


if __name__ == '__main__':
    unittest.main()
