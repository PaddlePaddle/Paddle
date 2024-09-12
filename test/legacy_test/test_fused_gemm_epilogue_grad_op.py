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

import os
import unittest

import numpy as np
from op_test import OpTest, skip_check_grad_ci

import paddle
from paddle.base import core


def is_rocm_gfx928():
    if not paddle.is_compiled_with_rocm():
        return False
    f = os.popen("$ROCM_PATH/bin/rocm_agent_enumerator | tail -1")
    if f.read()[:-1] == "gfx928":
        return True
    else:
        return False


def get_outputs(DOut, X, Y):
    DX = np.dot(DOut, Y.T)
    DY = np.dot(X.T, DOut)
    DBias = np.sum(DOut, axis=0)

    return DX, DY, DBias


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(
    not core.is_compiled_with_cuda() or not is_rocm_gfx928(),
    "core is not compiled with CUDA",
)
class TestFuseGemmEpilogueGradOpDXYBiasFP16(OpTest):
    def setUp(self):
        self.op_type = "fused_gemm_epilogue_grad"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.inputs = {
            'DOut': np.random.random((8, 128)).astype(self.dtype) - 0.5,
            'X': np.random.random((8, 4)).astype(self.dtype) - 0.5,
            'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5,
        }

        self.attrs = {"activation_grad": 'none'}

        DX, DY, DBias = get_outputs(
            self.inputs['DOut'], self.inputs['X'], self.inputs['Y']
        )
        self.outputs = {'DX': DX, 'DY': DY, 'DBias': DBias}

    def init_dtype_type(self):
        self.dtype = np.float16
        self.atol = 1e-3

    def test_check_output(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
            self.place
        ):
            return
        self.check_output_with_place(
            self.place, atol=self.atol, check_dygraph=False
        )


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(
    not core.is_compiled_with_cuda() or not is_rocm_gfx928(),
    "core is not compiled with CUDA",
)
class TestFuseGemmEpilogueGradOpDXYBiasFP32(
    TestFuseGemmEpilogueGradOpDXYBiasFP16
):
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(
    not core.is_compiled_with_cuda() or core.is_compiled_with_rocm(),
    "core is not compiled with CUDA or is compiled with ROCm",
)
class TestFuseGemmEpilogueGradOpDXYBiasFP64(
    TestFuseGemmEpilogueGradOpDXYBiasFP16
):
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(
    not core.is_compiled_with_cuda() or not is_rocm_gfx928(),
    "core is not compiled with CUDA",
)
class TestFuseGemmEpilogueGradOpDYBiasFP16(OpTest):
    def setUp(self):
        self.op_type = "fused_gemm_epilogue_grad"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.inputs = {
            'DOut': np.random.random((8, 128)).astype(self.dtype) - 0.5,
            'X': np.random.random((8, 4)).astype(self.dtype) - 0.5,
            'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5,
        }

        self.attrs = {"activation_grad": 'none'}

        _, DY, DBias = get_outputs(
            self.inputs['DOut'], self.inputs['X'], self.inputs['Y']
        )
        self.outputs = {'DY': DY, 'DBias': DBias}

    def init_dtype_type(self):
        self.dtype = np.float16
        self.atol = 1e-3

    def test_check_output(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
            self.place
        ):
            return
        self.check_output_with_place(
            self.place, atol=self.atol, check_dygraph=False
        )


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(
    not core.is_compiled_with_cuda() or not is_rocm_gfx928(),
    "core is not compiled with CUDA",
)
class TestFuseGemmEpilogueGradOpDYBiasFP32(
    TestFuseGemmEpilogueGradOpDYBiasFP16
):
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(
    not core.is_compiled_with_cuda() or core.is_compiled_with_rocm(),
    "core is not compiled with CUDA or is compiled with ROCm",
)
class TestFuseGemmEpilogueGradOpDYBiasFP64(
    TestFuseGemmEpilogueGradOpDYBiasFP16
):
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(
    not core.is_compiled_with_cuda() or not is_rocm_gfx928(),
    "core is not compiled with CUDA",
)
class TestFuseGemmEpilogueGradOpDYFP16(OpTest):
    def setUp(self):
        self.op_type = "fused_gemm_epilogue_grad"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.inputs = {
            'DOut': np.random.random((8, 128)).astype(self.dtype) - 0.5,
            'X': np.random.random((8, 4)).astype(self.dtype) - 0.5,
            'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5,
        }

        self.attrs = {"activation_grad": 'none'}

        _, DY, _ = get_outputs(
            self.inputs['DOut'], self.inputs['X'], self.inputs['Y']
        )
        self.outputs = {'DY': DY}

    def init_dtype_type(self):
        self.dtype = np.float16
        self.atol = 1e-3

    def test_check_output(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
            self.place
        ):
            return
        self.check_output_with_place(
            self.place, atol=self.atol, check_dygraph=False
        )


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(
    not core.is_compiled_with_cuda() or not is_rocm_gfx928(),
    "core is not compiled with CUDA",
)
class TestFuseGemmEpilogueGradOpDYFP32(TestFuseGemmEpilogueGradOpDYFP16):
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(
    not core.is_compiled_with_cuda() or core.is_compiled_with_rocm(),
    "core is not compiled with CUDA or is compiled with ROCm",
)
class TestFuseGemmEpilogueGradOpDYFP64(TestFuseGemmEpilogueGradOpDYFP16):
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(
    not core.is_compiled_with_cuda() or not is_rocm_gfx928(),
    "core is not compiled with CUDA",
)
class TestFuseGemmEpilogueGradOpDXYFP16(OpTest):
    def setUp(self):
        self.op_type = "fused_gemm_epilogue_grad"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.inputs = {
            'DOut': np.random.random((8, 128)).astype(self.dtype) - 0.5,
            'X': np.random.random((8, 4)).astype(self.dtype) - 0.5,
            'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5,
        }

        self.attrs = {"activation_grad": 'none'}

        DX, DY, _ = get_outputs(
            self.inputs['DOut'], self.inputs['X'], self.inputs['Y']
        )
        self.outputs = {'DX': DX, 'DY': DY}

    def init_dtype_type(self):
        self.dtype = np.float16
        self.atol = 1e-3

    def test_check_output(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
            self.place
        ):
            return
        self.check_output_with_place(
            self.place, atol=self.atol, check_dygraph=False
        )


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(
    not core.is_compiled_with_cuda() or not is_rocm_gfx928(),
    "core is not compiled with CUDA",
)
class TestFuseGemmEpilogueGradOpDXYFP32(TestFuseGemmEpilogueGradOpDXYFP16):
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(
    not core.is_compiled_with_cuda() or core.is_compiled_with_rocm(),
    "core is not compiled with CUDA or is compiled with ROCm",
)
class TestFuseGemmEpilogueGradOpDXYFP64(TestFuseGemmEpilogueGradOpDXYFP16):
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6


if __name__ == "__main__":
    paddle.enable_static()
    np.random.seed(0)
    unittest.main()
