// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <cublasXt.h>
#include <cublas_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <glog/logging.h>

/*
 * This file contains some CUDA specific utils.
 */

// For quickly implementing the prototype, some of the following code snippets
// are borrowed from project MXNet, great thanks for the original developers.

#define CHECK_CUDA_ERROR(msg)                                                \
  {                                                                          \
    auto e = cudaGetLastError();                                             \
    CHECK_EQ(e, cudaSuccess) << (msg) << " CUDA: " << cudaGetErrorString(e); \
  }

#define CUDA_CALL(func)                                      \
  {                                                          \
    auto e = (func);                                         \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) \
        << "CUDA: " << cudaGetErrorString(e);                \
  }

#define CUBLAS_CALL(func)                                        \
  {                                                              \
    auto e = (func);                                             \
    CHECK_EQ(e, CUBLAS_STATUS_SUCCESS)                           \
        << "cuBlas: " << paddle::lite::cuda::CublasErrorInfo(e); \
  }

namespace paddle {
namespace lite {
namespace cuda {

const char* CublasErrorInfo(int error) {
  switch (error) {
#define LITE_CUBLAS_ERROR_INFO(xx) \
  case xx:                         \
    return #xx;                    \
    break;
    LITE_CUBLAS_ERROR_INFO(CUBLAS_STATUS_NOT_INITIALIZED);
    LITE_CUBLAS_ERROR_INFO(CUBLAS_STATUS_ALLOC_FAILED);
    LITE_CUBLAS_ERROR_INFO(CUBLAS_STATUS_INVALID_VALUE);
    LITE_CUBLAS_ERROR_INFO(CUBLAS_STATUS_ARCH_MISMATCH);
    LITE_CUBLAS_ERROR_INFO(CUBLAS_STATUS_MAPPING_ERROR);
    LITE_CUBLAS_ERROR_INFO(CUBLAS_STATUS_EXECUTION_FAILED);
    LITE_CUBLAS_ERROR_INFO(CUBLAS_STATUS_INTERNAL_ERROR);
    LITE_CUBLAS_ERROR_INFO(CUBLAS_STATUS_NOT_SUPPORTED);
    LITE_CUBLAS_ERROR_INFO(CUBLAS_STATUS_LICENSE_ERROR);
#undef LITE_CUBLAS_ERROR_INFO
    default:
      return "unknown error";
  }
}

}  // namespace cuda
}  // namespace lite
}  // namespace paddle
