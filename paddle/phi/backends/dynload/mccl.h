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

#include <mccl.h>

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/backends/dynload/port.h"

namespace phi {
namespace dynload {

extern std::once_flag mccl_dso_flag;
extern void* mccl_dso_handle;

#define DECLARE_DYNAMIC_LOAD_MCCL_WRAP(__name)                   \
  struct DynLoad__##__name {                                     \
    template <typename... Args>                                  \
    auto operator()(Args... args) -> decltype(__name(args...)) { \
      using nccl_func = decltype(&::__name);                     \
      std::call_once(mccl_dso_flag, []() {                       \
        mccl_dso_handle = phi::dynload::GetNCCLDsoHandle();      \
      });                                                        \
      static void* p_##__name = dlsym(mccl_dso_handle, #__name); \
      return reinterpret_cast<nccl_func>(p_##__name)(args...);   \
    }                                                            \
  };                                                             \
  extern DynLoad__##__name __name

#define MCCL_RAND_ROUTINE_EACH(__macro) \
  __macro(mcclCommInitAll);             \
  __macro(mcclGetUniqueId);             \
  __macro(mcclCommInitRank);            \
  __macro(mcclCommAbort);               \
  __macro(mcclCommDestroy);             \
  __macro(mcclCommCount);               \
  __macro(mcclCommCuDevice);            \
  __macro(mcclCommUserRank);            \
  __macro(mcclAllReduce);               \
  __macro(mcclBcast);                   \
  __macro(mcclGroupStart);              \
  __macro(mcclAllGather);               \
  __macro(mcclGroupEnd);                \
  __macro(mcclReduce);                  \
  __macro(mcclReduceScatter);           \
  __macro(mcclCommGetAsyncError);       \
  __macro(mcclGetErrorString);

MCCL_RAND_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_MCCL_WRAP)

#define MCCL_RAND_ROUTINE_EACH_AFTER_2212(__macro) __macro(mcclBroadcast);
MCCL_RAND_ROUTINE_EACH_AFTER_2212(DECLARE_DYNAMIC_LOAD_MCCL_WRAP)

#define MCCL_RAND_ROUTINE_EACH_AFTER_2304(__macro) __macro(mcclGetVersion);
MCCL_RAND_ROUTINE_EACH_AFTER_2304(DECLARE_DYNAMIC_LOAD_MCCL_WRAP)

#define MCCL_RAND_ROUTINE_EACH_AFTER_2703(__macro) \
  __macro(mcclSend);                               \
  __macro(mcclRecv);
MCCL_RAND_ROUTINE_EACH_AFTER_2703(DECLARE_DYNAMIC_LOAD_MCCL_WRAP)

#define MCCL_RAND_ROUTINE_EACH_AFTER_21100(__macro) \
  __macro(mcclRedOpCreatePreMulSum);                \
  __macro(mcclRedOpDestroy);
MCCL_RAND_ROUTINE_EACH_AFTER_21100(DECLARE_DYNAMIC_LOAD_MCCL_WRAP)
}  // namespace dynload
}  // namespace phi
