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

#include <absl/container/flat_hash_map.h>
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>


#include "paddle/cinn/common/type.h"
#include "paddle/cinn/runtime/cinn_runtime.h"

namespace cinn {
namespace runtime {
namespace hip {

#define HIP_CALL(func)                                           \
  {                                                              \
    auto status = func;                                          \
    if (status != hipSuccess) {                                  \
      LOG(FATAL) << "HIP Error : " << hipGetErrorString(status); \
    }                                                            \
  }

#define HIP_DRIVER_CALL(func)                                                \
  {                                                                           \
    auto status = func;                                                       \
    if (status != hipSuccess) {                                               \
      const char* msg;                                                        \
      hipDrvGetErrorString(status, &msg);                                     \
      LOG(FATAL) << "HIP Driver Error: " #func " failed with error: " << msg; \
    }                                                                         \
  }

#define HIPRTC_CALL(func)                                             \
  {                                                                  \
    auto status = func;                                              \
    if (status != HIPRTC_SUCCESS) {                                   \
      LOG(FATAL) << "NVRTC Error : " << hiprtcGetErrorString(status); \
    }                                                                \
  }

const int kHIPMaxCards{8};

/**
 * Call a CUDA compiled kernel.
 *
 * @param kernel_fn the compiled kernel.
 * @param args an array of cinn_pod_value_ts(consists of scalars and buffers).
 */
void cinn_call_hip_kernel(void* kernel_fn,
                           void* v_args,
                           int num_args,
                           int grid_x,
                           int grid_y,
                           int grid_z,
                           int block_x,
                           int block_y,
                           int block_z,
                           void* stream);


}  // namespace hip
}  // namespace runtime
}  // namespace cinn
