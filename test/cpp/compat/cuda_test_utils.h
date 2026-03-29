// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "gtest/gtest.h"

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include <c10/cuda/CUDAFunctions.h>

#if defined(PADDLE_WITH_CUDA)
#include <cuda_runtime.h>
#elif defined(PADDLE_WITH_HIP)
#include <hip/hip_runtime.h>
#endif
#endif

namespace compat_test {

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
inline bool CudaRuntimeAvailable() {
  try {
    if (c10::cuda::device_count() <= 0) {
      return false;
    }
  } catch (...) {
    return false;
  }
#if defined(PADDLE_WITH_CUDA)
  return cudaFree(nullptr) == cudaSuccess;
#else
  return hipFree(nullptr) == hipSuccess;
#endif
}
#else
inline bool CudaRuntimeAvailable() { return false; }
#endif

}  // namespace compat_test

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#define SKIP_IF_CUDA_RUNTIME_UNAVAILABLE()      \
  do {                                          \
    if (!compat_test::CudaRuntimeAvailable()) { \
      return;                                   \
    }                                           \
  } while (false)
#else
#define SKIP_IF_CUDA_RUNTIME_UNAVAILABLE() \
  do {                                     \
  } while (false)
#endif
