// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#define HIP_CHECK(expr)                                               \
  {                                                                   \
    auto status = expr;                                               \
    if (status != hipSuccess) {                                       \
      PADDLE_THROW(phi::errors::Fatal("HIP Error in Paddle CINN: %s", \
                                      hipGetErrorString(status)));    \
    }                                                                 \
  }

#define HIP_DRIVER_CHECK(expr)                                         \
  {                                                                    \
    auto status = expr;                                                \
    if (status != hipSuccess) {                                        \
      const char* msg;                                                 \
      hipDrvGetErrorString(status, &msg);                              \
      PADDLE_THROW(phi::errors::Fatal(                                 \
          "HIP Driver Error in Paddle CINN: %s failed with error: %s", \
          #expr,                                                       \
          msg));                                                       \
    }                                                                  \
  }

#define HIPRTC_CHECK(expr)                                               \
  {                                                                      \
    auto status = expr;                                                  \
    if (status != HIPRTC_SUCCESS) {                                      \
      PADDLE_THROW(phi::errors::Fatal("HIPRTC Error in Paddle CINN: %s", \
                                      hiprtcGetErrorString(status)));    \
    }                                                                    \
  }
}  // namespace hip
}  // namespace runtime
}  // namespace cinn
