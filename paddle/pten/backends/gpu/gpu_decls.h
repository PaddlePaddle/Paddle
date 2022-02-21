// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/pten/backends/gpu/forwards.h"

namespace pten {

#ifdef PADDLE_WITH_HIP
#define DECLARE_TYPE_FOR_GPU(GPU_TYPE, CUDA_TYPE, ROCM_TYPE) \
  using GPU_TYPE = ROCM_TYPE;

#else  // PADDLE_WITH_CDUA

#define DECLARE_TYPE_FOR_GPU(GPU_TYPE, CUDA_TYPE, ROCM_TYPE) \
  using GPU_TYPE = CUDA_TYPE;
#endif

DECLARE_TYPE_FOR_GPU(gpuStream_t, cudaStream_t, hipStream_t);
DECLARE_TYPE_FOR_GPU(gpuEvent_t, cudaEvent_t, hipEvent_t);

DECLARE_TYPE_FOR_GPU(dnnActivationDescriptor,
                     cudnnActivationStruct,
                     miopenActivationDescriptor);
DECLARE_TYPE_FOR_GPU(dnnTensorDescriptor,
                     cudnnTensorStruct,
                     miopenTensorDescriptor);
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
DECLARE_TYPE_FOR_GPU(dnnDropoutDescriptor_t,
                     cudnnDropoutDescriptor_t,
                     miopenDropoutDescriptor_t);
DECLARE_TYPE_FOR_GPU(dnnHandle_t, cudnnHandle_t, miopenHandle_t);

DECLARE_TYPE_FOR_GPU(blasHandle_t, cublasHandle_t, rocblas_handle);

DECLARE_TYPE_FOR_GPU(solverHandle_t, cusolverDnHandle_t, rocsolver_handle);

DECLARE_TYPE_FOR_GPU(sparseHandle_t, cusparseHandle_t, rocsparse_handle);

#undef DECLARE_TYPE_FOR_GPU

using CUDAGraphID = unsigned long long;  // NOLINT

}  // namespace pten
