# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import annotations

import platform
import unittest
from typing import TYPE_CHECKING

import numpy as np
import tvm_ffi.cpp

import paddle
from paddle.utils.dlpack import DLDeviceType

if TYPE_CHECKING:
    from tvm_ffi import Module


class TestTVMFFIEnvStream(unittest.TestCase):
    def test_tvm_ffi_env_stream_for_gpu_tensor(self):
        if not paddle.is_compiled_with_cuda():
            return
        tensor = paddle.to_tensor([1.0, 2.0, 3.0]).cuda()
        current_raw_stream_ptr = tensor.__tvm_ffi_env_stream__()
        self.assertIsInstance(current_raw_stream_ptr, int)
        self.assertNotEqual(current_raw_stream_ptr, 0)

    def test_tvm_ffi_env_stream_for_cpu_tensor(self):
        tensor = paddle.to_tensor([1.0, 2.0, 3.0]).cpu()
        with self.assertRaisesRegex(
            RuntimeError, r"the __tvm_ffi_env_stream__ method"
        ):
            tensor.__tvm_ffi_env_stream__()


class TestCDLPackExchangeAPI(unittest.TestCase):
    def test_c_dlpack_exchange_api_cpu(self):
        cpp_source = r"""
            void add_one_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
                // implementation of a library function
                TVM_FFI_ICHECK(x.ndim() == 1) << "x must be a 1D tensor";
                DLDataType f32_dtype{kDLFloat, 32, 1};
                TVM_FFI_ICHECK(x.dtype() == f32_dtype) << "x must be a float tensor";
                TVM_FFI_ICHECK(y.ndim() == 1) << "y must be a 1D tensor";
                TVM_FFI_ICHECK(y.dtype() == f32_dtype) << "y must be a float tensor";
                TVM_FFI_ICHECK(x.size(0) == y.size(0)) << "x and y must have the same shape";
                for (int i = 0; i < x.size(0); ++i) {
                    static_cast<float*>(y.data_ptr())[i] = static_cast<float*>(x.data_ptr())[i] + 1;
                }
            }
        """

        mod: Module = tvm_ffi.cpp.load_inline(
            name='mod', cpp_sources=cpp_source, functions='add_one_cpu'
        )

        x = paddle.full((3,), 1.0, dtype='float32').cpu()
        y = paddle.zeros((3,), dtype='float32').cpu()
        mod.add_one_cpu(x, y)
        np.testing.assert_allclose(y.numpy(), [2.0, 2.0, 2.0])

    def test_c_dlpack_exchange_api_gpu(self):
        if not paddle.is_compiled_with_cuda():
            return
        if paddle.is_compiled_with_rocm():
            # Skip on DCU because CUDA_HOME is not available
            return
        if platform.system() == "Windows":
            # Temporary skip this test case on windows because compile bug on TVM FFI
            return
        cpp_sources = r"""
            void add_one_cuda(tvm::ffi::TensorView x, tvm::ffi::TensorView y);
        """
        cuda_sources = r"""
            __global__ void AddOneKernel(float* x, float* y, int n) {
              int idx = blockIdx.x * blockDim.x + threadIdx.x;
              if (idx < n) {
                y[idx] = x[idx] + 1;
              }
            }

            void add_one_cuda(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
              // implementation of a library function
              TVM_FFI_ICHECK(x.ndim() == 1) << "x must be a 1D tensor";
              DLDataType f32_dtype{kDLFloat, 32, 1};
              TVM_FFI_ICHECK(x.dtype() == f32_dtype) << "x must be a float tensor";
              TVM_FFI_ICHECK(y.ndim() == 1) << "y must be a 1D tensor";
              TVM_FFI_ICHECK(y.dtype() == f32_dtype) << "y must be a float tensor";
              TVM_FFI_ICHECK(x.size(0) == y.size(0)) << "x and y must have the same shape";

              int64_t n = x.size(0);
              int64_t nthread_per_block = 256;
              int64_t nblock = (n + nthread_per_block - 1) / nthread_per_block;
              // Obtain the current stream from the environment by calling TVMFFIEnvGetStream
              cudaStream_t stream = static_cast<cudaStream_t>(
                  TVMFFIEnvGetStream(x.device().device_type, x.device().device_id));
              // launch the kernel
              AddOneKernel<<<nblock, nthread_per_block, 0, stream>>>(static_cast<float*>(x.data_ptr()),
                                                                     static_cast<float*>(y.data_ptr()), n);
            }
        """
        mod: Module = tvm_ffi.cpp.load_inline(
            name='mod',
            cpp_sources=cpp_sources,
            cuda_sources=cuda_sources,
            functions=['add_one_cuda'],
        )

        x = paddle.full((3,), 1.0, dtype='float32').cuda()
        y = paddle.zeros((3,), dtype='float32').cuda()
        mod.add_one_cuda(x, y)
        np.testing.assert_allclose(y.numpy(), [2.0, 2.0, 2.0])

    def test_c_dlpack_exchange_api_alloc_tensor(self):
        cpp_source = r"""
            inline tvm::ffi::Tensor alloc_tensor(tvm::ffi::Shape shape, DLDataType dtype, DLDevice device) {
                return tvm::ffi::Tensor::FromEnvAlloc(TVMFFIEnvTensorAlloc, shape, dtype, device);
            }

            tvm::ffi::Tensor add_one_cpu(tvm::ffi::TensorView x) {
                TVM_FFI_ICHECK(x.ndim() == 1) << "x must be a 1D tensor";
                DLDataType f32_dtype{kDLFloat, 32, 1};
                TVM_FFI_ICHECK(x.dtype() == f32_dtype) << "x must be a float tensor";
                tvm::ffi::Tensor y = alloc_tensor(x.shape(), f32_dtype, x.device());
                for (int i = 0; i < x.size(0); ++i) {
                    static_cast<float*>(y.data_ptr())[i] = static_cast<float*>(x.data_ptr())[i] + 1;
                }
                return y;
            }
        """
        mod: Module = tvm_ffi.cpp.load_inline(
            name='mod', cpp_sources=cpp_source, functions=['add_one_cpu']
        )
        x = paddle.full((3,), 1.0, dtype='float32').cpu()
        y = mod.add_one_cpu(x)
        np.testing.assert_allclose(y.numpy(), [2.0, 2.0, 2.0])


class TestDLPackDataType(unittest.TestCase):
    @staticmethod
    def _paddle_dtype_to_tvm_ffi_dtype(paddle_dtype: paddle.dtype):
        dtype_str = str(paddle_dtype).split('.')[-1]
        return tvm_ffi.dtype(dtype_str)

    def test_dlpack_data_type_base_protocol(self):
        for dtype in [
            paddle.uint8,
            paddle.int16,
            paddle.int32,
            paddle.int64,
            paddle.float32,
            paddle.float64,
            paddle.float16,
            paddle.bfloat16,
        ]:
            tvm_ffi_dtype = TestDLPackDataType._paddle_dtype_to_tvm_ffi_dtype(
                dtype
            )
            self.assertEqual(
                dtype.__dlpack_data_type__(),
                (
                    tvm_ffi_dtype.type_code,
                    tvm_ffi_dtype.bits,
                    tvm_ffi_dtype.lanes,
                ),
            )

    # TODO(SigureMo): add e2e test case pass a paddle.dtype to TVM FFI Function
    # in tvm_ffi next release


class TestDLPackDeviceType(unittest.TestCase):
    def test_dlpack_device_type_base_protocol_from_place(self):
        self.assertEqual(
            paddle.CPUPlace().__dlpack_device__(),
            (DLDeviceType.kDLCPU.value, 0),
        )

        if paddle.is_compiled_with_cuda():
            self.assertEqual(
                paddle.CUDAPlace(0).__dlpack_device__(),
                (DLDeviceType.kDLCUDA.value, 0),
            )

            self.assertEqual(
                paddle.CUDAPinnedPlace().__dlpack_device__(),
                (DLDeviceType.kDLCUDAHost.value, 0),
            )

    def test_dlpack_device_type_base_protocol_from_device(self):
        self.assertEqual(
            paddle.device('cpu').__dlpack_device__(),
            (DLDeviceType.kDLCPU.value, 0),
        )

        if paddle.is_compiled_with_cuda():
            self.assertEqual(
                paddle.device('cuda:0').__dlpack_device__(),
                (DLDeviceType.kDLCUDA.value, 0),
            )

            self.assertEqual(
                paddle.device('gpu:0').__dlpack_device__(),
                (DLDeviceType.kDLCUDA.value, 0),
            )

    # TODO(SigureMo): add e2e test case pass a paddle.base.core.Place to TVM FFI Function
    # in tvm_ffi next release


if __name__ == '__main__':
    unittest.main()
