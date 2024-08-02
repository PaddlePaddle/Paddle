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
#include "paddle/common/enforce.h"

#define CUDA_DRIVER_CALL(func)                                         \
  {                                                                    \
    auto status = func;                                                \
    if (status != CUDA_SUCCESS) {                                      \
      const char* msg;                                                 \
      cuGetErrorString(status, &msg);                                  \
      std::stringstream ss;                                            \
      ss << "CUDA Driver Error: " #func " failed with error: " << msg; \
      PADDLE_THROW(::common::errors::Fatal(ss.str()));                 \
    }                                                                  \
  }

#define CUDA_CALL(func)                                    \
  {                                                        \
    auto status = func;                                    \
    if (status != cudaSuccess) {                           \
      std::stringstream ss;                                \
      ss << "CUDA Error : " << cudaGetErrorString(status); \
      PADDLE_THROW(::common::errors::Fatal(ss.str()));     \
    }                                                      \
  }

#define CURAND_CALL(func)                              \
  {                                                    \
    auto status = func;                                \
    if (status != CURAND_STATUS_SUCCESS) {             \
      std::stringstream ss;                            \
      ss << "CURAND Error : " << status;               \
      PADDLE_THROW(::common::errors::Fatal(ss.str())); \
    }                                                  \
  }

#define CUSOLVER_CALL(func)                            \
  {                                                    \
    auto status = func;                                \
    if (status != CUSOLVER_STATUS_SUCCESS) {           \
      std::stringstream ss;                            \
      ss << "CUSOLVER Error: " << status;              \
      PADDLE_THROW(::common::errors::Fatal(ss.str())); \
    }                                                  \
  }

#define CUBLAS_CALL(func)                                     \
  {                                                           \
    auto status = func;                                       \
    if (status != CUBLAS_STATUS_SUCCESS) {                    \
      PADDLE_THROW(::common::errors::Fatal("CUBLAS Error!")); \
    }                                                         \
  }

#define CUDNN_CALL(func)                                     \
  {                                                          \
    auto status = func;                                      \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::stringstream ss;                                  \
      ss << "CUDNN Error : " << cudnnGetErrorString(status); \
      PADDLE_THROW(::common::errors::Fatal(ss.str()));       \
    }                                                        \
  }

#define NVRTC_CALL(func)                                     \
  {                                                          \
    auto status = func;                                      \
    if (status != NVRTC_SUCCESS) {                           \
      std::stringstream ss;                                  \
      ss << "NVRTC Error : " << nvrtcGetErrorString(status); \
      PADDLE_THROW(::common::errors::Fatal(ss.str()));       \
    }                                                        \
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
