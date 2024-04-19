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

#include <string>

#include "paddle/phi/core/distributed/types.h"

#ifdef PADDLE_WITH_RCCL
#include <hip/hip_runtime.h>
#include "paddle/phi/backends/dynload/rccl.h"
#elif defined(PADDLE_WITH_MCCL)
#include <musa_runtime.h>
#include "paddle/phi/backends/dynload/mccl.h"
#else
#include <cuda_runtime.h>
#include "paddle/phi/backends/dynload/nccl.h"
#endif

namespace phi {
namespace distributed {

#define NCCL_CHECK(cmd)                                                \
  do {                                                                 \
    ncclResult_t r = cmd;                                              \
    if (r != mcclSuccess) {                                            \
      PADDLE_THROW(                                                    \
          phi::errors::External("Failed, NCCL error %s:%d '%s'\n",     \
                                __FILE__,                              \
                                __LINE__,                              \
                                phi::dynload::ncclGetErrorString(r))); \
    }                                                                  \
  } while (0)

#define MCCL_CHECK(cmd)                                                \
  do {                                                                 \
    mcclResult_t r = cmd;                                              \
    if (r != mcclSuccess) {                                            \
      PADDLE_THROW(                                                    \
          phi::errors::External("Failed, MCCL error %s:%d '%s'\n",     \
                                __FILE__,                              \
                                __LINE__,                              \
                                phi::dynload::mcclGetErrorString(r))); \
    }                                                                  \
  } while (0)

#ifdef PADDLE_WITH_NCCL
#define CUDA_CHECK(expr)                                                    \
  do {                                                                      \
    cudaError_t r = expr;                                                   \
    if (r != cudaSuccess) {                                                 \
      PADDLE_THROW(phi::errors::External("Failed, cuda error %s:%d '%s'\n", \
                                         __FILE__,                          \
                                         __LINE__,                          \
                                         cudaGetErrorString(r)));           \
    }                                                                       \
  } while (0)
#elif defined(PADDLE_WITH_MCCL)
#define MUSA_CHECK(expr)                                                    \
  do {                                                                      \
    musaError_t r = expr;                                                   \
    if (r != musaSuccess) {                                                 \
      PADDLE_THROW(phi::errors::External("Failed, musa error %s:%d '%s'\n", \
                                         __FILE__,                          \
                                         __LINE__,                          \
                                         musaGetErrorString(r)));           \
    }                                                                       \
  } while (0)
#else  // PADDLE_WITH_RCCL
#define HIP_CHECK(expr)                                                    \
  do {                                                                     \
    hipError_t r = expr;                                                   \
    if (r != hipSuccess) {                                                 \
      PADDLE_THROW(phi::errors::External("Failed, hip error %s:%d '%s'\n", \
                                         __FILE__,                         \
                                         __LINE__,                         \
                                         hipGetErrorString(r)));           \
    }                                                                      \
  } while (0)
#endif

mcclRedOp_t ToNCCLRedType(ReduceOp reduction);

std::string SerializeNCCLUniqueId(const mcclUniqueId& ncclID);

std::string NCCLDTypeToString(mcclDataType_t dtype);

std::string NCCLRedTypeToString(mcclRedOp_t op);

}  // namespace distributed
}  // namespace phi
