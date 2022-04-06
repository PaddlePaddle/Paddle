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


def gelu(x):
    y_ref = 0.5 * x * (
        1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))
    return y_ref.astype(x.dtype)


def relu(x):
    mask = x > 0
    return x * mask


def get_output(X, Y, bias, act):
    out = np.dot(X, Y) + bias
    if act == 'relu':
        return relu(out)
    elif act == 'gelu':
        return gelu(out)
    else:
        return out


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpReluMMFP16(OpTest):
    def setUp(self):
        self.op_type = "fused_gemm_epilogue"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.inputs = {
            'X': np.random.random((8, 4)).astype(self.dtype) - 0.5,
            'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5,
            'Bias': np.random.random((128, )).astype(self.dtype) - 0.5
        }
        self.outputs = {
            'Out': get_output(self.inputs['X'], self.inputs['Y'],
                              self.inputs['Bias'], 'relu')
        }
        self.attrs = {"activation": 'relu'}

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
class TestFuseGemmEpilogueOpReluMMFP32(TestFuseGemmEpilogueOpReluMMFP16):
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpReluMMFP64(TestFuseGemmEpilogueOpReluMMFP16):
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpReluMTMFP16(OpTest):
    def setUp(self):
        self.op_type = "fused_gemm_epilogue"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.inputs = {
            'X': np.random.random((4, 8)).astype(self.dtype) - 0.5,
            'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5,
            'Bias': np.random.random((128, )).astype(self.dtype) - 0.5
        }
        self.outputs = {
            'Out': get_output(self.inputs['X'].T, self.inputs['Y'],
                              self.inputs['Bias'], 'relu')
        }
        self.attrs = {'trans_x': True, "activation": 'relu'}

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
class TestFuseGemmEpilogueOpReluMTMFP32(TestFuseGemmEpilogueOpReluMTMFP16):
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpReluMTMFP64(TestFuseGemmEpilogueOpReluMTMFP16):
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpReluMMTFP16(OpTest):
    def setUp(self):
        self.op_type = "fused_gemm_epilogue"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.inputs = {
            'X': np.random.random((8, 4)).astype(self.dtype) - 0.5,
            'Y': np.random.random((128, 4)).astype(self.dtype) - 0.5,
            'Bias': np.random.random((128, )).astype(self.dtype) - 0.5
        }
        self.outputs = {
            'Out': get_output(self.inputs['X'], self.inputs['Y'].T,
                              self.inputs['Bias'], 'relu')
        }
        self.attrs = {'trans_y': True, "activation": 'relu'}

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
class TestFuseGemmEpilogueOpReluMMTFP32(TestFuseGemmEpilogueOpReluMMTFP16):
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpReluMMTFP64(TestFuseGemmEpilogueOpReluMMTFP16):
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpReluMTMTFP16(OpTest):
    def setUp(self):
        self.op_type = "fused_gemm_epilogue"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.inputs = {
            'X': np.random.random((4, 8)).astype(self.dtype) - 0.5,
            'Y': np.random.random((128, 4)).astype(self.dtype) - 0.5,
            'Bias': np.random.random((128, )).astype(self.dtype) - 0.5
        }
        self.outputs = {
            'Out': get_output(self.inputs['X'].T, self.inputs['Y'].T,
                              self.inputs['Bias'], 'relu')
        }
        self.attrs = {'trans_x': True, 'trans_y': True, "activation": 'relu'}

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
class TestFuseGemmEpilogueOpReluMTMTFP32(TestFuseGemmEpilogueOpReluMTMTFP16):
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpReluMTMTFP64(TestFuseGemmEpilogueOpReluMTMTFP16):
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpReluMMFP16MultiDimX(OpTest):
    def setUp(self):
        self.op_type = "fused_gemm_epilogue"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.inputs = {
            'X': np.random.random((2, 2, 8, 4)).astype(self.dtype) - 0.5,
            'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5,
            'Bias': np.random.random((128, )).astype(self.dtype) - 0.5
        }
        self.outputs = {
            'Out': get_output(self.inputs['X'].reshape(
                (-1, 4)), self.inputs['Y'], self.inputs['Bias'],
                              'relu').reshape((2, 2, 8, 128))
        }
        self.attrs = {"activation": 'relu'}

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
class TestFuseGemmEpilogueOpReluMMFP32MultiDimX(
        TestFuseGemmEpilogueOpReluMMFP16MultiDimX):
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpReluMMFP64MultiDimX(
        TestFuseGemmEpilogueOpReluMMFP16MultiDimX):
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpReluMTMFP16MultiDimX(OpTest):
    def setUp(self):
        self.op_type = "fused_gemm_epilogue"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.inputs = {
            'X': np.random.random((4, 2, 2, 8)).astype(self.dtype) - 0.5,
            'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5,
            'Bias': np.random.random((128, )).astype(self.dtype) - 0.5
        }
        self.outputs = {
            'Out': get_output(self.inputs['X'].reshape(
                (4, -1)).T, self.inputs['Y'], self.inputs['Bias'],
                              'relu').reshape((2, 2, 8, 128))
        }
        self.attrs = {'trans_x': True, "activation": 'relu'}

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
class TestFuseGemmEpilogueOpReluMTMFP32MultiDimX(
        TestFuseGemmEpilogueOpReluMTMFP16MultiDimX):
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpReluMTMFP64MultiDimX(
        TestFuseGemmEpilogueOpReluMTMFP16MultiDimX):
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpGeluMMFP16(OpTest):
    def setUp(self):
        self.op_type = "fused_gemm_epilogue"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.inputs = {
            'X': np.random.random((8, 4)).astype(self.dtype) - 0.5,
            'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5,
            'Bias': np.random.random((128, )).astype(self.dtype) - 0.5
        }

        self.attrs = {"activation": 'gelu'}

        self.outputs = {
            'Out': get_output(self.inputs['X'], self.inputs['Y'],
                              self.inputs['Bias'], 'gelu')
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
class TestFuseGemmEpilogueOpGeluMMFP32(TestFuseGemmEpilogueOpGeluMMFP16):
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpGeluMMFP64(TestFuseGemmEpilogueOpGeluMMFP16):
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpNoneMMFP16(OpTest):
    def setUp(self):
        self.op_type = "fused_gemm_epilogue"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.inputs = {
            'X': np.random.random((8, 4)).astype(self.dtype) - 0.5,
            'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5,
            'Bias': np.random.random((128, )).astype(self.dtype) - 0.5
        }

        self.attrs = {"activation": 'none'}

        self.outputs = {
            'Out': get_output(self.inputs['X'], self.inputs['Y'],
                              self.inputs['Bias'], 'none')
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
class TestFuseGemmEpilogueOpNoneMMFP32(TestFuseGemmEpilogueOpNoneMMFP16):
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpNoneMMFP64(TestFuseGemmEpilogueOpNoneMMFP16):
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6


if __name__ == "__main__":
    np.random.seed(0)
    unittest.main()
