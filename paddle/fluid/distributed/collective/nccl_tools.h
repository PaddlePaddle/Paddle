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

#include "paddle/fluid/distributed/collective/types.h"

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

namespace paddle {
namespace distributed {

#define NCCL_CHECK(cmd)                            \
  do {                                             \
    ncclResult_t r = cmd;                          \
    if (r != ncclSuccess) {                        \
      printf("Failed, NCCL error %s:%d '%s'\n",    \
             __FILE__,                             \
             __LINE__,                             \
             phi::dynload::ncclGetErrorString(r)); \
      exit(EXIT_FAILURE);                          \
    }                                              \
  } while (0)

ncclRedOp_t ToNCCLRedType(ReduceOp reduction);

std::string SerializeNCCLUniqueId(const ncclUniqueId& ncclID);

}  // namespace distributed
}  // namespace paddle
