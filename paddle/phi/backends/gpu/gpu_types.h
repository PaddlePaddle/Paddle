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

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

#ifdef PADDLE_WITH_HIP
#include "paddle/phi/backends/dynload/miopen.h"
#include "paddle/phi/backends/dynload/rocblas.h"
#else  // PADDLE_WITH_CUDA
#include "paddle/phi/backends/dynload/cublas.h"
#include "paddle/phi/backends/dynload/cudnn.h"
#endif

namespace phi {

#ifdef PADDLE_WITH_HIP
#define DECLARE_TYPE_FOR_GPU(GPU_TYPE, CUDA_TYPE, ROCM_TYPE) \
  using GPU_TYPE = ROCM_TYPE;

#else  // PADDLE_WITH_CDUA

#define DECLARE_TYPE_FOR_GPU(GPU_TYPE, CUDA_TYPE, ROCM_TYPE) \
  using GPU_TYPE = CUDA_TYPE;
#endif

DECLARE_TYPE_FOR_GPU(gpuError_t, cudaError_t, hipError_t);
DECLARE_TYPE_FOR_GPU(gpuMemcpyKind, cudaMemcpyKind, hipMemcpyKind);
DECLARE_TYPE_FOR_GPU(gpuDeviceProp, cudaDeviceProp, hipDeviceProp_t);
DECLARE_TYPE_FOR_GPU(dnnDataType_t, cudnnDataType_t, miopenDataType_t);
DECLARE_TYPE_FOR_GPU(dnnPoolingMode_t, cudnnPoolingMode_t, miopenPoolingMode_t);
DECLARE_TYPE_FOR_GPU(dnnTensorFormat_t,
                     cudnnTensorFormat_t,
                     miopenTensorFormat_t);
DECLARE_TYPE_FOR_GPU(dnnActivationMode_t,
                     cudnnActivationMode_t,
                     miopenActivationMode_t);

#undef DECLARE_TYPE_FOR_GPU

#ifdef PADDLE_WITH_HIP
#define DECLARE_CONSTANT_FOR_GPU(GPU_CV, CUDA_CV, ROCM_CV) \
  constexpr auto GPU_CV = ROCM_CV;
#else  // PADDLE_WITH_CUDA
#define DECLARE_CONSTANT_FOR_GPU(GPU_CV, CUDA_CV, ROCM_CV) \
  constexpr auto GPU_CV = CUDA_CV;
#endif

DECLARE_CONSTANT_FOR_GPU(gpuErrorOutOfMemory,
                         cudaErrorMemoryAllocation,
                         hipErrorOutOfMemory);
DECLARE_CONSTANT_FOR_GPU(gpuErrorNotReady, cudaErrorNotReady, hipErrorNotReady);
DECLARE_CONSTANT_FOR_GPU(gpuSuccess, cudaSuccess, hipSuccess);

#undef DECLARE_CONSTANT_FOR_GPU
}  // namespace phi

#endif  // defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
