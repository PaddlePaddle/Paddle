// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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
#ifdef CINN_WITH_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <curand.h>
#include <glog/logging.h>

#include <string>
#include <tuple>
#include <vector>

#include "paddle/cinn/runtime/cinn_runtime.h"

#define CUDA_DRIVER_CALL(func)                                                 \
  {                                                                            \
    auto status = func;                                                        \
    if (status != CUDA_SUCCESS) {                                              \
      const char* msg;                                                         \
      cuGetErrorString(status, &msg);                                          \
      LOG(FATAL) << "CUDA Driver Error: " #func " failed with error: " << msg; \
    }                                                                          \
  }

#define CUDA_CALL(func)                                            \
  {                                                                \
    auto status = func;                                            \
    if (status != cudaSuccess) {                                   \
      LOG(FATAL) << "CUDA Error : " << cudaGetErrorString(status); \
    }                                                              \
  }

#define CURAND_CALL(func)                        \
  {                                              \
    auto status = func;                          \
    if (status != CURAND_STATUS_SUCCESS) {       \
      LOG(FATAL) << "CURAND Error : " << status; \
    }                                            \
  }

#define CUSOLVER_CALL(func)                       \
  {                                               \
    auto status = func;                           \
    if (status != CUSOLVER_STATUS_SUCCESS) {      \
      LOG(FATAL) << "CUSOLVER Error: " << status; \
    }                                             \
  }

#define CUBLAS_CALL(func)                  \
  {                                        \
    auto status = func;                    \
    if (status != CUBLAS_STATUS_SUCCESS) { \
      LOG(FATAL) << "CUBLAS Error!";       \
    }                                      \
  }

#define CUDNN_CALL(func)                                             \
  {                                                                  \
    auto status = func;                                              \
    if (status != CUDNN_STATUS_SUCCESS) {                            \
      LOG(FATAL) << "CUDNN Error : " << cudnnGetErrorString(status); \
    }                                                                \
  }

#define NVRTC_CALL(func)                                             \
  {                                                                  \
    auto status = func;                                              \
    if (status != NVRTC_SUCCESS) {                                   \
      LOG(FATAL) << "NVRTC Error : " << nvrtcGetErrorString(status); \
    }                                                                \
  }

namespace cinn {
namespace backends {

// CUDA syntax for thread axis.
std::string cuda_thread_axis_name(int level);

// CUDA syntax for block axis.
std::string cuda_block_axis_name(int level);

}  // namespace backends
}  // namespace cinn

#endif  // CINN_WITH_CUDA
