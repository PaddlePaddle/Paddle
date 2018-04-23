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

import paddle.fluid.core as core
from op_test import OpTest


def max_pool2D_forward_naive(x,
                             ksize,
                             strides,
                             paddings,
                             global_pool=0,
                             ceil_mode=False):
    N, C, H, W = x.shape
    if global_pool == 1:
        ksize = [H, W]
    H_out = (H - ksize[0] + 2 * paddings[0] + strides[0] - 1
             ) / strides[0] + 1 if ceil_mode else (H - ksize[0] + 2 *
                                                   paddings[0]) / strides[0] + 1
    W_out = (W - ksize[1] + 2 * paddings[1] + strides[1] - 1
             ) / strides[1] + 1 if ceil_mode else (W - ksize[1] + 2 *
                                                   paddings[1]) / strides[1] + 1
    out = np.zeros((N, C, H_out, W_out))
    for i in xrange(H_out):
        for j in xrange(W_out):
            r_start = np.max((i * strides[0] - paddings[0], 0))
            r_end = np.min((i * strides[0] + ksize[0] - paddings[0], H))
            c_start = np.max((j * strides[1] - paddings[1], 0))
            c_end = np.min((j * strides[1] + ksize[1] - paddings[1], W))
            x_masked = x[:, :, r_start:r_end, c_start:c_end]

            out[:, :, i, j] = np.max(x_masked, axis=(2, 3))
    return out


def avg_pool2D_forward_naive(x,
                             ksize,
                             strides,
                             paddings,
                             global_pool=0,
                             ceil_mode=False):
    N, C, H, W = x.shape
    if global_pool == 1:
        ksize = [H, W]
    H_out = (H - ksize[0] + 2 * paddings[0] + strides[0] - 1
             ) / strides[0] + 1 if ceil_mode else (H - ksize[0] + 2 *
                                                   paddings[0]) / strides[0] + 1
    W_out = (W - ksize[1] + 2 * paddings[1] + strides[1] - 1
             ) / strides[1] + 1 if ceil_mode else (W - ksize[1] + 2 *
                                                   paddings[1]) / strides[1] + 1
    out = np.zeros((N, C, H_out, W_out))
    for i in xrange(H_out):
        for j in xrange(W_out):
            r_start = np.max((i * strides[0] - paddings[0], 0))
            r_end = np.min((i * strides[0] + ksize[0] - paddings[0], H))
            c_start = np.max((j * strides[1] - paddings[1], 0))
            c_end = np.min((j * strides[1] + ksize[1] - paddings[1], W))
            x_masked = x[:, :, r_start:r_end, c_start:c_end]

            out[:, :, i, j] = np.sum(x_masked, axis=(2, 3)) / (
                (r_end - r_start) * (c_end - c_start))
    return out


class TestPool2d_Op(OpTest):
    def setUp(self):
        self.op_type = "pool2d"
        self.use_cudnn = False
        self.use_mkldnn = False
        self.dtype = np.float32
        self.init_test_case()
        self.init_global_pool()
        self.init_kernel_type()
        self.init_pool_type()
        self.init_ceil_mode()
        if self.global_pool:
            self.paddings = [0 for _ in range(len(self.paddings))]
        input = np.random.random(self.shape).astype(self.dtype)
        output = self.pool2D_forward_naive(input, self.ksize, self.strides,
                                           self.paddings, self.global_pool,
                                           self.ceil_mode).astype(self.dtype)
        self.inputs = {'X': OpTest.np_dtype_to_fluid_dtype(input)}

        self.attrs = {
            'strides': self.strides,
            'paddings': self.paddings,
            'ksize': self.ksize,
            'pooling_type': self.pool_type,
            'global_pooling': self.global_pool,
            'use_cudnn': self.use_cudnn,
            'use_mkldnn': self.use_mkldnn,
            'ceil_mode': self.ceil_mode,
            'data_format': 'AnyLayout'  # TODO(dzhwinter) : should be fix latter
        }

        self.outputs = {'Out': output}

    def testcudnn(self):
        return core.is_compiled_with_cuda() and self.use_cudnn

    def test_check_output(self):
        if self.testcudnn():
            place = core.CUDAPlace(0)
            self.check_output_with_place(place, atol=1e-5)
        else:
            self.check_output()

    def test_check_grad(self):
        if self.dtype == np.float16:
            return
        if self.testcudnn() and self.pool_type != "max":
            place = core.CUDAPlace(0)
            self.check_grad_with_place(
                place, set(['X']), 'Out', max_relative_error=0.07)
        elif self.pool_type != "max":
            self.check_grad(set(['X']), 'Out', max_relative_error=0.07)

    def init_test_case(self):
        self.shape = [2, 3, 5, 5]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [0, 0]

    def init_kernel_type(self):
        pass

    def init_pool_type(self):
        self.pool_type = "avg"
        self.pool2D_forward_naive = avg_pool2D_forward_naive

    def init_global_pool(self):
        self.global_pool = True

    def init_ceil_mode(self):
        self.ceil_mode = False


class TestCase1(TestPool2d_Op):
    def init_test_case(self):
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [0, 0]

    def init_pool_type(self):
        self.pool_type = "avg"
        self.pool2D_forward_naive = avg_pool2D_forward_naive

    def init_global_pool(self):
        self.global_pool = False


class TestCase2(TestPool2d_Op):
    def init_test_case(self):
        self.shape = [2, 3, 7, 7]
        self.ksize = [3, 3]
        self.strides = [1, 1]
        self.paddings = [1, 1]

    def init_pool_type(self):
        self.pool_type = "avg"
        self.pool2D_forward_naive = avg_pool2D_forward_naive

    def init_global_pool(self):
        self.global_pool = False


class TestCase3(TestPool2d_Op):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive


class TestCase4(TestCase1):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive


class TestCase5(TestCase2):
    def init_pool_type(self):
        self.pool_type = "max"
        self.pool2D_forward_naive = max_pool2D_forward_naive


#--------------------test pool2d--------------------
class TestCUDNNCase1(TestPool2d_Op):
    def init_kernel_type(self):
        self.use_cudnn = True


class TestFP16CUDNNCase1(TestPool2d_Op):
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


if __name__ == '__main__':
    unittest.main()
