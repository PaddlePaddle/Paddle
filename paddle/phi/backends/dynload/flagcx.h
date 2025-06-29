/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
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

#include <flagcx.h>

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/port.h"

namespace phi {
namespace dynload {

extern std::once_flag flagcx_dso_flag;
extern void* flagcx_dso_handle;

#define DECLARE_DYNAMIC_LOAD_FLAGCX_WRAP(__name)                     \
  struct DynLoad__##__name {                                         \
    template <typename... Args>                                      \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { \
      using flagcx_func = decltype(&::__name);                       \
      std::call_once(flagcx_dso_flag, []() {                         \
        flagcx_dso_handle = phi::dynload::GetFLAGCXDsoHandle();      \
      });                                                            \
      static void* p_##__name = dlsym(flagcx_dso_handle, #__name);   \
      return reinterpret_cast<flagcx_func>(p_##__name)(args...);     \
    }                                                                \
  };                                                                 \
  extern struct DynLoad__##__name __name

#define FLAGCX_RAND_ROUTINE_EACH(__macro) \
  __macro(flagcxGetUniqueId);             \
  __macro(flagcxCommInitRank);            \
  __macro(flagcxGetVersion);              \
  __macro(flagcxCommAbort);               \
  __macro(flagcxCommDestroy);             \
  __macro(flagcxCommCount);               \
  __macro(flagcxCommUserRank);            \
  __macro(flagcxAllReduce);               \
  __macro(flagcxBroadcast);               \
  __macro(flagcxAllGather);               \
  __macro(flagcxGroupStart);              \
  __macro(flagcxGroupEnd);                \
  __macro(flagcxReduce);                  \
  __macro(flagcxReduceScatter);           \
  __macro(flagcxCommGetAsyncError);       \
  __macro(flagcxSend);                    \
  __macro(flagcxRecv);                    \
  __macro(flagcxHandleInit);              \
  __macro(flagcxHandleFree);              \
  __macro(flagcxGetErrorString);

FLAGCX_RAND_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_FLAGCX_WRAP)

#undef DECLARE_DYNAMIC_LOAD_FLAGCX_WRAP

}  // namespace dynload
}  // namespace phi
