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

<<<<<<< HEAD
import unittest

import numpy as np
from op_test import OpTest, skip_check_grad_ci

import paddle
import paddle.fluid.core as core
=======
from __future__ import print_function

import unittest
import numpy as np
import paddle
import paddle.fluid.core as core
from op_test import OpTest, skip_check_grad_ci
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81


def get_outputs(DOut, X, Y):
    DX = np.dot(DOut, Y.T)
    DY = np.dot(X.T, DOut)
    DBias = np.sum(DOut, axis=0)

    return DX, DY, DBias


@skip_check_grad_ci(reason="no grap op")
<<<<<<< HEAD
@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFuseGemmEpilogueGradOpDXYBiasFP16(OpTest):
=======
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueGradOpDXYBiasFP16(OpTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "fused_gemm_epilogue_grad"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.inputs = {
            'DOut': np.random.random((8, 128)).astype(self.dtype) - 0.5,
            'X': np.random.random((8, 4)).astype(self.dtype) - 0.5,
<<<<<<< HEAD
            'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5,
=======
            'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }

        self.attrs = {"activation": 'none'}

<<<<<<< HEAD
        DX, DY, DBias = get_outputs(
            self.inputs['DOut'], self.inputs['X'], self.inputs['Y']
        )
=======
        DX, DY, DBias = get_outputs(self.inputs['DOut'], self.inputs['X'],
                                    self.inputs['Y'])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.outputs = {'DX': DX, 'DY': DY, 'DBias': DBias}

    def init_dtype_type(self):
        self.dtype = np.float16
        self.atol = 1e-3

    def test_check_output(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
<<<<<<< HEAD
            self.place
        ):
=======
                self.place):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            return
        self.check_output_with_place(self.place, atol=self.atol)


@skip_check_grad_ci(reason="no grap op")
<<<<<<< HEAD
@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFuseGemmEpilogueGradOpDXYBiasFP32(
    TestFuseGemmEpilogueGradOpDXYBiasFP16
):
=======
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueGradOpDXYBiasFP32(
        TestFuseGemmEpilogueGradOpDXYBiasFP16):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
<<<<<<< HEAD
@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFuseGemmEpilogueGradOpDXYBiasFP64(
    TestFuseGemmEpilogueGradOpDXYBiasFP16
):
=======
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueGradOpDXYBiasFP64(
        TestFuseGemmEpilogueGradOpDXYBiasFP16):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
<<<<<<< HEAD
@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFuseGemmEpilogueGradOpDYBiasFP16(OpTest):
=======
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueGradOpDYBiasFP16(OpTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "fused_gemm_epilogue_grad"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.inputs = {
            'DOut': np.random.random((8, 128)).astype(self.dtype) - 0.5,
            'X': np.random.random((8, 4)).astype(self.dtype) - 0.5,
<<<<<<< HEAD
            'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5,
=======
            'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }

        self.attrs = {"activation": 'none'}

<<<<<<< HEAD
        _, DY, DBias = get_outputs(
            self.inputs['DOut'], self.inputs['X'], self.inputs['Y']
        )
=======
        _, DY, DBias = get_outputs(self.inputs['DOut'], self.inputs['X'],
                                   self.inputs['Y'])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.outputs = {'DY': DY, 'DBias': DBias}

    def init_dtype_type(self):
        self.dtype = np.float16
        self.atol = 1e-3

    def test_check_output(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
<<<<<<< HEAD
            self.place
        ):
=======
                self.place):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            return
        self.check_output_with_place(self.place, atol=self.atol)


@skip_check_grad_ci(reason="no grap op")
<<<<<<< HEAD
@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFuseGemmEpilogueGradOpDYBiasFP32(
    TestFuseGemmEpilogueGradOpDYBiasFP16
):
=======
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueGradOpDYBiasFP32(TestFuseGemmEpilogueGradOpDYBiasFP16
                                           ):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
<<<<<<< HEAD
@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFuseGemmEpilogueGradOpDYBiasFP64(
    TestFuseGemmEpilogueGradOpDYBiasFP16
):
=======
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueGradOpDYBiasFP64(TestFuseGemmEpilogueGradOpDYBiasFP16
                                           ):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
<<<<<<< HEAD
@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFuseGemmEpilogueGradOpDYFP16(OpTest):
=======
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueGradOpDYFP16(OpTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "fused_gemm_epilogue_grad"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.inputs = {
            'DOut': np.random.random((8, 128)).astype(self.dtype) - 0.5,
            'X': np.random.random((8, 4)).astype(self.dtype) - 0.5,
<<<<<<< HEAD
            'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5,
=======
            'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }

        self.attrs = {"activation": 'none'}

<<<<<<< HEAD
        _, DY, _ = get_outputs(
            self.inputs['DOut'], self.inputs['X'], self.inputs['Y']
        )
=======
        _, DY, _ = get_outputs(self.inputs['DOut'], self.inputs['X'],
                               self.inputs['Y'])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.outputs = {'DY': DY}

    def init_dtype_type(self):
        self.dtype = np.float16
        self.atol = 1e-3

    def test_check_output(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
<<<<<<< HEAD
            self.place
        ):
=======
                self.place):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            return
        self.check_output_with_place(self.place, atol=self.atol)


@skip_check_grad_ci(reason="no grap op")
<<<<<<< HEAD
@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFuseGemmEpilogueGradOpDYFP32(TestFuseGemmEpilogueGradOpDYFP16):
=======
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueGradOpDYFP32(TestFuseGemmEpilogueGradOpDYFP16):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
<<<<<<< HEAD
@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFuseGemmEpilogueGradOpDYFP64(TestFuseGemmEpilogueGradOpDYFP16):
=======
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueGradOpDYFP64(TestFuseGemmEpilogueGradOpDYFP16):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
<<<<<<< HEAD
@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFuseGemmEpilogueGradOpDXYFP16(OpTest):
=======
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueGradOpDXYFP16(OpTest):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def setUp(self):
        self.op_type = "fused_gemm_epilogue_grad"
        self.place = core.CUDAPlace(0)
        self.init_dtype_type()

        self.inputs = {
            'DOut': np.random.random((8, 128)).astype(self.dtype) - 0.5,
            'X': np.random.random((8, 4)).astype(self.dtype) - 0.5,
<<<<<<< HEAD
            'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5,
=======
            'Y': np.random.random((4, 128)).astype(self.dtype) - 0.5
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        }

        self.attrs = {"activation": 'none'}

<<<<<<< HEAD
        DX, DY, _ = get_outputs(
            self.inputs['DOut'], self.inputs['X'], self.inputs['Y']
        )
=======
        DX, DY, _ = get_outputs(self.inputs['DOut'], self.inputs['X'],
                                self.inputs['Y'])
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        self.outputs = {'DX': DX, 'DY': DY}

    def init_dtype_type(self):
        self.dtype = np.float16
        self.atol = 1e-3

    def test_check_output(self):
        if self.dtype == np.float16 and not core.is_float16_supported(
<<<<<<< HEAD
            self.place
        ):
=======
                self.place):
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            return
        self.check_output_with_place(self.place, atol=self.atol)


@skip_check_grad_ci(reason="no grap op")
<<<<<<< HEAD
@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFuseGemmEpilogueGradOpDXYFP32(TestFuseGemmEpilogueGradOpDXYFP16):
=======
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueGradOpDXYFP32(TestFuseGemmEpilogueGradOpDXYFP16):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype_type(self):
        self.dtype = np.single
        self.atol = 1e-6


@skip_check_grad_ci(reason="no grap op")
<<<<<<< HEAD
@unittest.skipIf(
    not core.is_compiled_with_cuda(), "core is not compiled with CUDA"
)
class TestFuseGemmEpilogueGradOpDXYFP64(TestFuseGemmEpilogueGradOpDXYFP16):
=======
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueGradOpDXYFP64(TestFuseGemmEpilogueGradOpDXYFP16):

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    def init_dtype_type(self):
        self.dtype = np.double
        self.atol = 1e-6


if __name__ == "__main__":
    paddle.enable_static()
    np.random.seed(0)
    unittest.main()
