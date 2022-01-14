# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
# Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
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

import unittest
import numpy as np
import paddle
import paddle.fluid.core as core
from op_test import OpTest, skip_check_grad_ci


def get_output(X, Y, bias, act='relu'):
    out = np.dot(X, Y) + bias
    if (act == 'relu'):
        mask = out > 0
        return out * mask
    else:
        return out


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpMMFP16(OpTest):
    def setUp(self):
        self.op_type = "fused_gemm_epilogue"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.inputs = {
            'X': np.random.random((8, 4)).astype(self.dtype) - 0.5,
            'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5,
            'bias': np.random.random((128, )).astype(self.dtype) - 0.5
        }
        self.outputs = {
            'out': get_output(self.inputs['X'], self.inputs['Y'],
                              self.inputs['bias'])
        }

    def init_dtype_type(self):
        self.dtype = np.float16
        self.atol = 1e-3

    def test_check_output(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
                self.place):
            return
        self.check_output_with_place(self.place, atol=self.atol)


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpMMFP32(TestFuseGemmEpilogueOpMMFP16):
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpMMFP64(TestFuseGemmEpilogueOpMMFP16):
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpMTMFP16(OpTest):
    def setUp(self):
        self.op_type = "fused_gemm_epilogue"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.inputs = {
            'X': np.random.random((4, 8)).astype(self.dtype) - 0.5,
            'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5,
            'bias': np.random.random((128, )).astype(self.dtype) - 0.5
        }
        self.outputs = {
            'out': get_output(self.inputs['X'].T, self.inputs['Y'],
                              self.inputs['bias'])
        }
        self.attrs = {'trans_x': True}

    def init_dtype_type(self):
        self.dtype = np.float16
        self.atol = 1e-3

    def test_check_output(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
                self.place):
            return
        self.check_output_with_place(self.place, atol=self.atol)


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpMTMFP32(TestFuseGemmEpilogueOpMTMFP16):
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpMTMFP64(TestFuseGemmEpilogueOpMTMFP16):
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpMMTFP16(OpTest):
    def setUp(self):
        self.op_type = "fused_gemm_epilogue"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.inputs = {
            'X': np.random.random((8, 4)).astype(self.dtype) - 0.5,
            'Y': np.random.random((128, 4)).astype(self.dtype) - 0.5,
            'bias': np.random.random((128, )).astype(self.dtype) - 0.5
        }
        self.outputs = {
            'out': get_output(self.inputs['X'], self.inputs['Y'].T,
                              self.inputs['bias'])
        }
        self.attrs = {'trans_y': True}

    def init_dtype_type(self):
        self.dtype = np.float16
        self.atol = 1e-3

    def test_check_output(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
                self.place):
            return
        self.check_output_with_place(self.place, atol=self.atol)


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpMMTFP32(TestFuseGemmEpilogueOpMMTFP16):
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpMMTFP64(TestFuseGemmEpilogueOpMMTFP16):
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpMTMTFP16(OpTest):
    def setUp(self):
        self.op_type = "fused_gemm_epilogue"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.inputs = {
            'X': np.random.random((4, 8)).astype(self.dtype) - 0.5,
            'Y': np.random.random((128, 4)).astype(self.dtype) - 0.5,
            'bias': np.random.random((128, )).astype(self.dtype) - 0.5
        }
        self.outputs = {
            'out': get_output(self.inputs['X'].T, self.inputs['Y'].T,
                              self.inputs['bias'])
        }
        self.attrs = {'trans_x': True, 'trans_y': True}

    def init_dtype_type(self):
        self.dtype = np.float16
        self.atol = 1e-3

    def test_check_output(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
                self.place):
            return
        self.check_output_with_place(self.place, atol=self.atol)


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpMTMTFP32(TestFuseGemmEpilogueOpMTMTFP16):
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpMTMTFP64(TestFuseGemmEpilogueOpMTMTFP16):
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpMMFP16MultiDimX(OpTest):
    def setUp(self):
        self.op_type = "fused_gemm_epilogue"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.inputs = {
            'X': np.random.random((2, 2, 8, 4)).astype(self.dtype) - 0.5,
            'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5,
            'bias': np.random.random((128, )).astype(self.dtype) - 0.5
        }
        self.outputs = {
            'out': get_output(self.inputs['X'].reshape((-1, 4)),
                              self.inputs['Y'], self.inputs['bias']).reshape(
                                  (2, 2, 8, 128))
        }

    def init_dtype_type(self):
        self.dtype = np.float16
        self.atol = 1e-3

    def test_check_output(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
                self.place):
            return
        self.check_output_with_place(self.place, atol=self.atol)


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpMMFP32MultiDimX(
        TestFuseGemmEpilogueOpMMFP16MultiDimX):
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpMMFP64MultiDimX(
        TestFuseGemmEpilogueOpMMFP16MultiDimX):
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpMTMFP16MultiDimX(OpTest):
    def setUp(self):
        self.op_type = "fused_gemm_epilogue"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.inputs = {
            'X': np.random.random((4, 2, 2, 8)).astype(self.dtype) - 0.5,
            'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5,
            'bias': np.random.random((128, )).astype(self.dtype) - 0.5
        }
        self.outputs = {
            'out': get_output(self.inputs['X'].reshape((4, -1)).T,
                              self.inputs['Y'], self.inputs['bias']).reshape(
                                  (2, 2, 8, 128))
        }
        self.attrs = {'trans_x': True}

    def init_dtype_type(self):
        self.dtype = np.float16
        self.atol = 1e-3

    def test_check_output(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
                self.place):
            return
        self.check_output_with_place(self.place, atol=self.atol)


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpMTMFP32MultiDimX(
        TestFuseGemmEpilogueOpMTMFP16MultiDimX):
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpMTMFP64MultiDimX(
        TestFuseGemmEpilogueOpMTMFP16MultiDimX):
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6


if __name__ == "__main__":
    unittest.main()
