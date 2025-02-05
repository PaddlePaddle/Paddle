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

import sys
import unittest

import numpy as np
from op_test import _set_use_system_allocator
from test_batch_norm_op_prim_nchw import TestBatchNormOp

import paddle
from paddle.base import core

paddle.enable_static()

np.random.seed(123)
paddle.seed(123)

_set_use_system_allocator(True)


class TestBatchNormOpNHWCTestMode(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-5
        self.fw_comp_rtol = 1e-5
        self.rev_comp_atol = 1e-5
        self.rev_comp_rtol = 1e-5
        self.dtype = "float32"
        self.shape = [16, 16, 16, 8]
        self.training = False
        self.momentum = 0.1
        self.epsilon = 1e-05
        self.data_format = "NHWC"
        self.use_global_stats = True
        self.check_cpu_prim_pir_grad = True


class TestBatchNormOpNHWCTestModeFp64(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-15
        self.fw_comp_rtol = 1e-15
        self.rev_comp_atol = 1e-15
        self.rev_comp_rtol = 1e-15
        self.dtype = "float64"
        self.shape = [16, 16, 16, 8]
        self.training = False
        self.momentum = 0.1
        self.epsilon = 1e-05
        self.data_format = "NHWC"
        self.use_global_stats = None


class TestBatchNormOpNHWCTestModeFp16(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-3
        self.fw_comp_rtol = 1e-3
        self.rev_comp_atol = 1e-3
        self.rev_comp_rtol = 1e-3
        self.dtype = "float16"
        self.shape = [16, 16, 16, 8]
        self.training = False
        self.momentum = 0.1
        self.epsilon = 1e-05
        self.data_format = "NHWC"
        self.use_global_stats = None


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestBatchNormOpNHWCTestModebf16(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-3
        self.fw_comp_rtol = 1e-3
        self.rev_comp_atol = 1e-3
        self.rev_comp_rtol = 1e-3
        # prim bf16 has diff in windows
        if sys.platform == "win32":
            self.rev_comp_atol = 5e-3
            self.rev_comp_rtol = 5e-3
        self.cinn_atol = 1e-3
        self.cinn_rtol = 1e-3
        self.dtype = "uint16"
        self.shape = [16, 16, 16, 8]
        self.training = False
        self.momentum = 0.1
        self.epsilon = 1e-05
        self.data_format = "NHWC"
        self.use_global_stats = None


class TestBatchNormOpNHWC(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-5
        self.fw_comp_rtol = 1e-5
        self.rev_comp_atol = 1e-5
        self.rev_comp_rtol = 1e-5
        self.dtype = "float32"
        self.shape = [16, 16, 16, 8]
        self.training = True
        self.momentum = 0.1
        self.epsilon = 1e-05
        self.data_format = "NHWC"
        self.use_global_stats = None


class TestBatchNormOpNHWCFp64(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-11
        self.fw_comp_rtol = 1e-11
        self.rev_comp_atol = 1e-11
        self.rev_comp_rtol = 1e-11
        self.dtype = "float64"
        self.shape = [16, 16, 16, 8]
        self.training = True
        self.momentum = 0.1
        self.epsilon = 1e-05
        self.data_format = "NHWC"
        self.use_global_stats = None
        self.check_prim_pir = True
        self.check_prim_pir_grad = True
        self.check_cpu_prim_pir_grad = True


class TestBatchNormOpNHWCFp16(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-3
        self.fw_comp_rtol = 1e-3
        self.rev_comp_atol = 1e-3
        self.rev_comp_rtol = 1e-3
        self.dtype = "float16"
        self.shape = [16, 16, 16, 8]
        self.training = True
        self.momentum = 0.1
        self.epsilon = 1e-05
        self.data_format = "NHWC"
        self.use_global_stats = None


@unittest.skipIf(
    not core.is_compiled_with_cuda()
    or not core.is_bfloat16_supported(core.CUDAPlace(0)),
    "core is not compiled with CUDA or not support the bfloat16",
)
class TestBatchNormOpNHWCbf16(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-3
        self.fw_comp_rtol = 1e-3
        self.rev_comp_atol = 1e-3
        self.rev_comp_rtol = 1e-3
        # prim bf16 has diff in windows
        if sys.platform == "win32":
            self.rev_comp_atol = 5e-3
            self.rev_comp_rtol = 5e-3
        self.cinn_atol = 1e-3
        self.cinn_rtol = 1e-3
        self.dtype = "uint16"
        self.shape = [16, 16, 16, 8]
        self.training = True
        self.momentum = 0.1
        self.epsilon = 1e-05
        self.data_format = "NHWC"
        self.use_global_stats = None


class TestBatchNormOpNHWCShape2(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-5
        self.fw_comp_rtol = 1e-5
        self.rev_comp_atol = 1e-5
        self.rev_comp_rtol = 1e-5
        self.dtype = "float32"
        self.shape = [4, 8, 16, 32]
        self.training = True
        self.momentum = 0.1
        self.epsilon = 1e-05
        self.data_format = "NHWC"
        self.use_global_stats = None


class TestBatchNormOpNHWCMomentum2(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-5
        self.fw_comp_rtol = 1e-5
        self.rev_comp_atol = 1e-5
        self.rev_comp_rtol = 1e-5
        self.dtype = "float32"
        self.shape = [16, 16, 16, 8]
        self.training = True
        self.momentum = 0.9
        self.epsilon = 1e-05
        self.data_format = "NHWC"
        self.use_global_stats = None


class TestBatchNormOpNHWCEps2(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-5
        self.fw_comp_rtol = 1e-5
        self.rev_comp_atol = 1e-5
        self.rev_comp_rtol = 1e-5
        self.dtype = "float32"
        self.shape = [16, 16, 16, 8]
        self.training = True
        self.momentum = 0.1
        self.epsilon = 1e-06
        self.data_format = "NHWC"
        self.use_global_stats = None


class TestBatchNormOpNHWCShape3(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-5
        self.fw_comp_rtol = 1e-5
        self.rev_comp_atol = 1e-5
        self.rev_comp_rtol = 1e-5
        self.dtype = "float32"
        self.shape = [4, 128, 32]
        self.training = True
        self.momentum = 0.1
        self.epsilon = 1e-05
        self.data_format = "NHWC"
        self.use_global_stats = None


class TestBatchNormOpNHWCShape4(TestBatchNormOp):
    def initConfig(self):
        self.fw_comp_atol = 1e-5
        self.fw_comp_rtol = 1e-5
        self.rev_comp_atol = 1e-5
        self.rev_comp_rtol = 1e-5
        self.dtype = "float32"
        self.shape = [4, 256]
        self.training = True
        self.momentum = 0.1
        self.epsilon = 1e-05
        self.data_format = "NHWC"
        self.use_global_stats = None


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
