// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/backends/gpu/forwards.h"
#include "paddle/phi/backends/gpu/gpu_decls.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) || \
    defined(PADDLE_WITH_MUSA)

#ifdef PADDLE_WITH_HIP
#include "paddle/phi/backends/dynload/miopen.h"
#include "paddle/phi/backends/dynload/rocblas.h"
#elif defined(PADDLE_WITH_MUSA)
#include "paddle/phi/backends/dynload/mublas.h"
#include "paddle/phi/backends/dynload/mudnn.h"
#else  // PADDLE_WITH_CUDA
#include "paddle/phi/backends/dynload/cublas.h"
#include "paddle/phi/backends/dynload/cudnn.h"
#endif

namespace phi {

#ifdef PADDLE_WITH_HIP
#define DECLARE_TYPE_FOR_GPU(GPU_TYPE, CUDA_TYPE, ROCM_TYPE, MUSA_TYPE) \
  using GPU_TYPE = ROCM_TYPE;

#elif defined(PADDLE_WITH_MUSA)
#define DECLARE_TYPE_FOR_GPU(GPU_TYPE, CUDA_TYPE, ROCM_TYPE, MUSA_TYPE) \
  using GPU_TYPE = MUSA_TYPE;

#else  // PADDLE_WITH_MUSA
#define DECLARE_TYPE_FOR_GPU(GPU_TYPE, CUDA_TYPE, ROCM_TYPE, MUSA_TYPE) \
  using GPU_TYPE = CUDA_TYPE;
#endif  // PADDLE_WITH_CUDA

DECLARE_TYPE_FOR_GPU(gpuError_t, cudaError_t, hipError_t, musaError_t);
DECLARE_TYPE_FOR_GPU(gpuMemcpyKind,
                     cudaMemcpyKind,
                     hipMemcpyKind,
                     musaMemcpyKind);
DECLARE_TYPE_FOR_GPU(gpuDeviceProp,
                     cudaDeviceProp,
                     hipDeviceProp_t,
                     musaDeviceProp);
#undef DECLARE_TYPE_FOR_GPU

#ifndef PADDLE_WITH_MUSA
#ifdef PADDLE_WITH_HIP
#define DECLARE_TYPE_FOR_GPU(GPU_TYPE, CUDA_TYPE, ROCM_TYPE) \
  using GPU_TYPE = ROCM_TYPE;

#else  // PADDLE_WITH_MUSA
#define DECLARE_TYPE_FOR_GPU(GPU_TYPE, CUDA_TYPE, ROCM_TYPE) \
  using GPU_TYPE = CUDA_TYPE;
#endif  // PADDLE_WITH_CUDA

DECLARE_TYPE_FOR_GPU(dnnDataType_t, cudnnDataType_t, miopenDataType_t);
DECLARE_TYPE_FOR_GPU(dnnPoolingMode_t, cudnnPoolingMode_t, miopenPoolingMode_t);
DECLARE_TYPE_FOR_GPU(dnnTensorFormat_t,
                     cudnnTensorFormat_t,
                     miopenTensorFormat_t);
DECLARE_TYPE_FOR_GPU(dnnActivationMode_t,
                     cudnnActivationMode_t,
                     miopenActivationMode_t);
#undef DECLARE_TYPE_FOR_GPU
#endif

#ifdef PADDLE_WITH_HIP
#define DECLARE_CONSTANT_FOR_GPU(GPU_CV, CUDA_CV, ROCM_CV, MUSA_CV) \
  constexpr auto GPU_CV = ROCM_CV;
#elif defined(PADDLE_WITH_MUSA)
#define DECLARE_CONSTANT_FOR_GPU(GPU_CV, CUDA_CV, ROCM_CV, MUSA_CV) \
  constexpr auto GPU_CV = MUSA_CV;
#else  // PADDLE_WITH_CUDA
#define DECLARE_CONSTANT_FOR_GPU(GPU_CV, CUDA_CV, ROCM_CV, MUSA_CV) \
  constexpr auto GPU_CV = CUDA_CV;
#endif

DECLARE_CONSTANT_FOR_GPU(gpuErrorOutOfMemory,
                         cudaErrorMemoryAllocation,
                         hipErrorOutOfMemory,
                         musaErrorMemoryAllocation);
DECLARE_CONSTANT_FOR_GPU(gpuErrorNotReady,
                         cudaErrorNotReady,
                         hipErrorNotReady,
                         musaErrorNotReady);
DECLARE_CONSTANT_FOR_GPU(gpuSuccess, cudaSuccess, hipSuccess, musaSuccess);

DECLARE_CONSTANT_FOR_GPU(gpuMemcpyHostToDevice,
                         cudaMemcpyKind::cudaMemcpyHostToDevice,
                         hipMemcpyKind::hipMemcpyHostToDevice,
                         musaMemcpyKind::musaMemcpyHostToDevice);
DECLARE_CONSTANT_FOR_GPU(gpuMemcpyDeviceToHost,
                         cudaMemcpyKind::cudaMemcpyDeviceToHost,
                         hipMemcpyKind::hipMemcpyDeviceToHost,
                         musaMemcpyKind::musaMemcpyDeviceToHost);
DECLARE_CONSTANT_FOR_GPU(gpuMemcpyDeviceToDevice,
                         cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                         hipMemcpyKind::hipMemcpyDeviceToDevice,
                         musaMemcpyKind::musaMemcpyDeviceToDevice);

#undef DECLARE_CONSTANT_FOR_GPU
}  // namespace phi

#endif  // defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP) ||
        // defined(PADDLE_WITH_MUSA )
