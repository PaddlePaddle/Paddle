// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
// Copyright (c) 2022 NVIDIA Corporation. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || defined(PADDLE_WITH_MUSA)

#ifdef PADDLE_WITH_HIP
#include <hip/hip_runtime.h>

#include "paddle/fluid/platform/dynload/miopen.h"
#include "paddle/fluid/platform/dynload/rocblas.h"

#elif defined(PADDLE_WITH_MUSA)
#include <musa_runtime.h>
#include <mublas.h>
#include <mudnn.h>
using mudnnHandle_t = class Handle*;
//TODO(Xiaokang Shang)
#else
#include <cuda_runtime.h>

#include "paddle/fluid/platform/dynload/cublas.h"
#include "paddle/fluid/platform/dynload/cublasLt.h"
#include "paddle/fluid/platform/dynload/cudnn.h"
#endif

namespace paddle {

#ifdef PADDLE_WITH_HIP
#define DECLARE_TYPE_FOR_GPU(GPU_TYPE, CUDA_TYPE, ROCM_TYPE, MUSA_TYPE) \
  using GPU_TYPE = ROCM_TYPE;

#elif defined(PADDLE_WITH_MUSA)
#define DECLARE_TYPE_FOR_GPU(GPU_TYPE, CUDA_TYPE, ROCM_TYPE, MUSA_TYPE) \
  using GPU_TYPE = MUSA_TYPE;
#else  // CDUA

#define DECLARE_TYPE_FOR_GPU(GPU_TYPE, CUDA_TYPE, ROCM_TYPE, MUSA_TYPE) \
  using GPU_TYPE = CUDA_TYPE;
#endif

DECLARE_TYPE_FOR_GPU(gpuStream_t, cudaStream_t, hipStream_t, musaStream_t);
DECLARE_TYPE_FOR_GPU(gpuError_t, cudaError_t, hipError_t, musaError_t);
DECLARE_TYPE_FOR_GPU(gpuEvent_t, cudaEvent_t, hipEvent_t, musaEvent_t);
DECLARE_TYPE_FOR_GPU(gpuMemcpyKind, cudaMemcpyKind, hipMemcpyKind, musaMemcpyKind);
DECLARE_TYPE_FOR_GPU(gpuDeviceProp, cudaDeviceProp, hipDeviceProp_t, musaDeviceProp);

// TODO(Xiaokang Shang): confirm mudnn type
#if 0
DECLARE_TYPE_FOR_GPU(dnnDataType_t, cudnnDataType_t, miopenDataType_t, mudnnDataType_t);
DECLARE_TYPE_FOR_GPU(dnnActivationDescriptor,
                     cudnnActivationStruct,
                     miopenActivationDescriptor,
                     mudnnActivationStruct);
DECLARE_TYPE_FOR_GPU(dnnActivationMode_t,
                     cudnnActivationMode_t,
                     miopenActivationMode_t);
DECLARE_TYPE_FOR_GPU(dnnTensorDescriptor,
                     cudnnTensorStruct,
                     miopenTensorDescriptor);
DECLARE_TYPE_FOR_GPU(dnnTensorFormat_t,
                     cudnnTensorFormat_t,
                     miopenTensorFormat_t);
DECLARE_TYPE_FOR_GPU(dnnFilterDescriptor,
                     cudnnFilterStruct,
                     miopenTensorDescriptor);
DECLARE_TYPE_FOR_GPU(dnnFilterDescriptor_t,
                     cudnnFilterDescriptor_t,
                     miopenTensorDescriptor_t);
DECLARE_TYPE_FOR_GPU(dnnConvolutionDescriptor,
                     cudnnConvolutionStruct,
                     miopenConvolutionDescriptor);
DECLARE_TYPE_FOR_GPU(dnnConvolutionDescriptor_t,
                     cudnnConvolutionDescriptor_t,
                     miopenConvolutionDescriptor_t);
DECLARE_TYPE_FOR_GPU(dnnPoolingDescriptor_t,
                     cudnnPoolingDescriptor_t,
                     miopenPoolingDescriptor_t);
DECLARE_TYPE_FOR_GPU(dnnPoolingMode_t, cudnnPoolingMode_t, miopenPoolingMode_t);
DECLARE_TYPE_FOR_GPU(dnnDropoutDescriptor_t,
                     cudnnDropoutDescriptor_t,
                     miopenDropoutDescriptor_t);
#endif

DECLARE_TYPE_FOR_GPU(blasLtHandle_t, cublasLtHandle_t, rocblas_handle, mublasHandle_t); // TODO(MTAI)
DECLARE_TYPE_FOR_GPU(dnnHandle_t, cudnnHandle_t, miopenHandle_t, mudnnHandle_t);
DECLARE_TYPE_FOR_GPU(blasHandle_t, cublasHandle_t, rocblas_handle, mublasHandle_t);

// TODO(Ming Huang): Since there is no blasLt handler,
// use rocblas_handle for workround.
//DECLARE_TYPE_FOR_GPU(blasLtHandle_t, cublasLtHandle_t, rocblas_handle);

using CUDAGraphID = unsigned long long;  // NOLINT

#undef DECLARE_TYPE_FOR_GPU

#if defined(PADDLE_WITH_HIP)
#define DECLARE_CONSTANT_FOR_GPU(GPU_CV, CUDA_CV, ROCM_CV, MUSA_CV) \
  constexpr auto GPU_CV = ROCM_CV;
#elif defined(PADDLE_WITH_MUSA)
#define DECLARE_CONSTANT_FOR_GPU(GPU_CV, CUDA_CV, ROCM_CV, MUSA_CV) \
  constexpr auto GPU_CV = MUSA_CV;
#else  // CDUA

#define DECLARE_CONSTANT_FOR_GPU(GPU_CV, CUDA_CV, ROCM_CV, MUSA_CV) \
  constexpr auto GPU_CV = CUDA_CV;
#endif

DECLARE_CONSTANT_FOR_GPU(gpuErrorOutOfMemory,
                         cudaErrorMemoryAllocation,
                         hipErrorOutOfMemory,
                         musaErrorMemoryAllocation);
DECLARE_CONSTANT_FOR_GPU(gpuErrorNotReady, cudaErrorNotReady, hipErrorNotReady, musaErrorNotReady);
DECLARE_CONSTANT_FOR_GPU(gpuSuccess, cudaSuccess, hipSuccess, musaSuccess);

#undef DECLARE_CONSTANT_FOR_GPU
}  // namespace paddle

#endif
