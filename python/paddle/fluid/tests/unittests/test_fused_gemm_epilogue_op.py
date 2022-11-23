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

import unittest
import numpy as np
import paddle
import paddle.fluid.core as core
from op_test import OpTest, skip_check_grad_ci, skip_check_inplace_ci


def is_fused_gemm_epilogue_supported():
    if paddle.is_compiled_with_cuda() and not paddle.is_compiled_with_rocm():
        return hasattr(core.eager.ops, 'fused_gemm_epilogue')
    else:
        return False


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


@skip_check_inplace_ci(reason="no inplace op")
class TestFuseGemmBase(OpTest):
    pass


@skip_check_grad_ci(reason="no grap op")
@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFuseGemmEpilogueOpReluMMFP16(TestFuseGemmBase):

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
            'Out':
            get_output(self.inputs['X'], self.inputs['Y'], self.inputs['Bias'],
                       'relu')
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
class TestFuseGemmEpilogueOpReluMTMFP16(TestFuseGemmBase):

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
            'Out':
            get_output(self.inputs['X'].T, self.inputs['Y'],
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
class TestFuseGemmEpilogueOpReluMMTFP16(TestFuseGemmBase):

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
            'Out':
            get_output(self.inputs['X'], self.inputs['Y'].T,
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
class TestFuseGemmEpilogueOpReluMTMTFP16(TestFuseGemmBase):

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
            'Out':
            get_output(self.inputs['X'].T, self.inputs['Y'].T,
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
class TestFuseGemmEpilogueOpReluMMFP16MultiDimX(TestFuseGemmBase):

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
            'Out':
            get_output(self.inputs['X'].reshape((-1, 4)), self.inputs['Y'],
                       self.inputs['Bias'], 'relu').reshape((2, 2, 8, 128))
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
class TestFuseGemmEpilogueOpReluMTMFP16MultiDimX(TestFuseGemmBase):

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
            'Out':
            get_output(self.inputs['X'].reshape((4, -1)).T, self.inputs['Y'],
                       self.inputs['Bias'], 'relu').reshape((2, 2, 8, 128))
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
class TestFuseGemmEpilogueOpGeluMMFP16(TestFuseGemmBase):

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
            'Out':
            get_output(self.inputs['X'], self.inputs['Y'], self.inputs['Bias'],
                       'gelu')
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
class TestFuseGemmEpilogueOpNoneMMFP16(TestFuseGemmBase):

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
            'Out':
            get_output(self.inputs['X'], self.inputs['Y'], self.inputs['Bias'],
                       'none')
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


def matmul(x, y, bias, trans_x, trans_y):
    x = np.array(x)
    if trans_x:
        x = np.ascontiguousarray(np.transpose(x))
    if trans_y:
        y = np.ascontiguousarray(np.transpose(y))
    z = np.matmul(x, y)
    if bias is None:
        return z
    else:
        return z + bias


def matmul_grad(x, y, bias, dz, trans_x, trans_y):
    if trans_x:
        if trans_y:
            dx = matmul(y, dz, None, True, True)
            dy = matmul(dz, x, None, True, True)
        else:
            dx = matmul(y, dz, None, False, True)
            dy = matmul(x, dz, None, False, False)
    else:
        if trans_y:
            dx = matmul(dz, y, None, False, False)
            dy = matmul(dz, x, None, True, False)
        else:
            dx = matmul(dz, y, None, False, True)
            dy = matmul(x, dz, None, True, False)
    if bias is None:
        dbias = None
    else:
        dbias = np.sum(dz, axis=0, keepdims=False)
    return dx, dy, dbias


@unittest.skipIf(
    not is_fused_gemm_epilogue_supported(),
    "fused_gemm_epilogue is only supported when CUDA version >= 11.6")
class TestEagerFusedGemmEpilogue(unittest.TestCase):

    def setUp(self):
        paddle.set_device('gpu')

    def test_case_act(self):
        paddle.disable_static()
        x_np = np.random.random((8, 4)).astype(np.float64) - 0.5
        y_np = np.random.random((4, 128)).astype(np.float64) - 0.5
        bias_np = np.random.random((128, )).astype(np.float64) - 0.5
        x = paddle.to_tensor(x_np)
        y = paddle.to_tensor(y_np)
        bias = paddle.to_tensor(bias_np)
        x.stop_gradient = False
        y.stop_gradient = False

        out1 = core.eager.ops.fused_gemm_epilogue(x, y, bias, 'trans_x', False,
                                                  'trans_y', False,
                                                  'activation', 'none')
        out2 = core.eager.ops.fused_gemm_epilogue(x, y, bias, 'trans_x', False,
                                                  'trans_y', False,
                                                  'activation', 'relu')
        out3 = core.eager.ops.fused_gemm_epilogue(x, y, bias, 'trans_x', False,
                                                  'trans_y', False,
                                                  'activation', 'gelu')

        out_np1 = get_output(x_np, y_np, bias_np, 'none')
        out_np2 = get_output(x_np, y_np, bias_np, 'relu')
        out_np3 = get_output(x_np, y_np, bias_np, 'gelu')

        np.testing.assert_allclose(out1, out_np1, rtol=1e-05)
        np.testing.assert_allclose(out2, out_np2, rtol=1e-05)
        np.testing.assert_allclose(out3, out_np3, rtol=1e-05)

        out_grad_np1 = np.random.randint(low=-20, high=20,
                                         size=out_np1.shape).astype(np.float64)
        paddle.autograd.backward(out1,
                                 grad_tensors=[paddle.to_tensor(out_grad_np1)])

        x_grad_np, y_grad_np, bias_grad_np = matmul_grad(
            x_np, y_np, bias_np, out_grad_np1, False, False)
        np.testing.assert_allclose(x.grad.numpy(), x_grad_np, rtol=1e-05)
        self.assertEqual(y_grad_np.shape, y_np.shape)
        np.testing.assert_allclose(y.grad.numpy(), y_grad_np, rtol=1e-05)

        paddle.enable_static()


if __name__ == "__main__":
    paddle.enable_static()
    np.random.seed(0)
    unittest.main()
