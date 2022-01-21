/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once

#include <rccl.h>

#include <mutex>  // NOLINT
#include "paddle/pten/backends/dynload/rccl.h"

namespace paddle {
namespace platform {
namespace dynload {

#define PLATFORM_DECLARE_DYNAMIC_LOAD_RCCL_WRAP(__name)       \
  using DynLoad__##__name = pten::dynload::DynLoad__##__name; \
  extern DynLoad__##__name __name

#define RCCL_RAND_ROUTINE_EACH(__macro) \
  __macro(ncclCommInitAll);             \
  __macro(ncclGetUniqueId);             \
  __macro(ncclCommInitRank);            \
  __macro(ncclCommDestroy);             \
  __macro(ncclCommCount);               \
  __macro(ncclCommCuDevice);            \
  __macro(ncclCommUserRank);            \
  __macro(ncclAllReduce);               \
  __macro(ncclBcast);                   \
  __macro(ncclAllGather);               \
  __macro(ncclGroupStart);              \
  __macro(ncclGroupEnd);                \
  __macro(ncclReduce);                  \
  __macro(ncclReduceScatter);           \
  __macro(ncclGetErrorString);

RCCL_RAND_ROUTINE_EACH(PLATFORM_DECLARE_DYNAMIC_LOAD_RCCL_WRAP)

#if NCCL_VERSION_CODE >= 2212
#define RCCL_RAND_ROUTINE_EACH_AFTER_2212(__macro) __macro(ncclBroadcast);
RCCL_RAND_ROUTINE_EACH_AFTER_2212(PLATFORM_DECLARE_DYNAMIC_LOAD_RCCL_WRAP)
#endif

#if NCCL_VERSION_CODE >= 2703
#define RCCL_RAND_ROUTINE_EACH_AFTER_2703(__macro) \
  __macro(ncclSend);                               \
  __macro(ncclRecv);
RCCL_RAND_ROUTINE_EACH_AFTER_2703(PLATFORM_DECLARE_DYNAMIC_LOAD_RCCL_WRAP)
#endif

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
