// Copyright (c) 2024 CINN Authors. All Rights Reserved.
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

#ifdef CINN_WITH_XPU
#include <cuda.h>
#include <cuda_runtime_api.h>
#endif

#include "paddle/cinn/runtime/cinn_runtime.h"
#include "paddle/common/enforce.h"

namespace cinn {
namespace runtime {
namespace xpu {

// ---------------------------------------------------------------------------
// Error-check macros
// ---------------------------------------------------------------------------

#define XPU_CHECK(expr)                                                      \
  {                                                                          \
    auto status = (expr);                                                    \
    if (status != cudaSuccess) {                                             \
      PADDLE_THROW(::common::errors::Fatal("XPU (CUDA RT) Error in Paddle " \
                                           "CINN: %s",                      \
                                           cudaGetErrorString(status)));     \
    }                                                                        \
  }

#define XPU_DRIVER_CHECK(expr)                                                \
  {                                                                           \
    CUresult status = (expr);                                                 \
    if (status != CUDA_SUCCESS) {                                             \
      const char* msg = nullptr;                                              \
      cuGetErrorString(status, &msg);                                         \
      PADDLE_THROW(::common::errors::Fatal(                                   \
          "XPU (CUDA Driver) Error in Paddle CINN: %s failed with error: %s", \
          #expr,                                                               \
          msg ? msg : "unknown"));                                             \
    }                                                                         \
  }

// ---------------------------------------------------------------------------
// Host-callable kernel launcher
// ---------------------------------------------------------------------------

/**
 * Launch an XPU (CUDA) kernel via the CUDA driver API.
 *
 * @param kernel_fn          CUfunction handle cast to void*.
 * @param v_args             Pointer to array of cinn_pod_value_t arguments.
 * @param num_args           Number of arguments.
 * @param grid_x/y/z         Grid dimensions.
 * @param block_x/y/z        Block dimensions.
 * @param shared_memory_bytes Shared memory in bytes.
 * @param stream             cudaStream_t cast to void* (may be nullptr).
 */
void cinn_call_xpu_kernel(void* kernel_fn,
                          void* v_args,
                          int num_args,
                          int grid_x,
                          int grid_y,
                          int grid_z,
                          int block_x,
                          int block_y,
                          int block_z,
                          int shared_memory_bytes,
                          void* stream);

void infer_shape_set_value(int row, int col, int64_t value, int64_t** v);

}  // namespace xpu
}  // namespace runtime
}  // namespace cinn
