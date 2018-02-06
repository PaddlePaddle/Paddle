/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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

#include <dlfcn.h>
#include <nccl.h>
#include <mutex>
#include "paddle/platform/call_once.h"
#include "paddle/platform/dynload/dynamic_loader.h"

namespace paddle {
namespace platform {
namespace dynload {

extern std::once_flag nccl_dso_flag;
extern void* nccl_dso_handle;

#ifdef PADDLE_USE_DSO
#define DECLARE_DYNAMIC_LOAD_NCCL_WRAP(__name)                         \
  struct DynLoad__##__name {                                           \
    template <typename... Args>                                        \
    auto operator()(Args... args) -> decltype(__name(args...)) {       \
      using nccl_func = decltype(__name(args...)) (*)(Args...);        \
      platform::call_once(nccl_dso_flag,                               \
                          paddle::platform::dynload::GetNCCLDsoHandle, \
                          &nccl_dso_handle);                           \
      void* p_##__name = dlsym(nccl_dso_handle, #__name);              \
      return reinterpret_cast<nccl_func>(p_##__name)(args...);         \
    }                                                                  \
  };                                                                   \
  extern DynLoad__##__name __name
#else
#define DECLARE_DYNAMIC_LOAD_NCCL_WRAP(__name) \
  struct DynLoad__##__name {                   \
    template <typename... Args>                \
    ncclResult_t operator()(Args... args) {    \
      return __name(args...);                  \
    }                                          \
  };                                           \
  extern DynLoad__##__name __name
#endif

#define NCCL_RAND_ROUTINE_EACH(__macro) \
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
  __macro(ncclReduce);                  \
  __macro(ncclGetErrorString);

NCCL_RAND_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_NCCL_WRAP)

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
