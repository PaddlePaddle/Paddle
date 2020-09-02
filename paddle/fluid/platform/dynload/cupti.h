/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#ifdef PADDLE_WITH_CUPTI

#include <cuda.h>
#include <cupti.h>
#include <mutex>  // NOLINT

#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/port.h"

namespace paddle {
namespace platform {
namespace dynload {

extern std::once_flag cupti_dso_flag;
extern void *cupti_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load cupti routine
 * via operator overloading.
 *
 * note: default dynamic linked libs
 */
#define DECLARE_DYNAMIC_LOAD_CUPTI_WRAP(__name)                            \
  struct DynLoad__##__name {                                               \
    template <typename... Args>                                            \
    inline CUptiResult CUPTIAPI operator()(Args... args) {                 \
      using cuptiFunc = decltype(&::__name);                               \
      std::call_once(cupti_dso_flag, []() {                                \
        cupti_dso_handle = paddle::platform::dynload::GetCUPTIDsoHandle(); \
      });                                                                  \
      static void *p_##__name = dlsym(cupti_dso_handle, #__name);          \
      return reinterpret_cast<cuptiFunc>(p_##__name)(args...);             \
    }                                                                      \
  };                                                                       \
  extern DynLoad__##__name __name

#define CUPTI_ROUTINE_EACH(__macro)           \
  __macro(cuptiActivityEnable);               \
  __macro(cuptiActivityDisable);              \
  __macro(cuptiActivityRegisterCallbacks);    \
  __macro(cuptiActivityGetAttribute);         \
  __macro(cuptiActivitySetAttribute);         \
  __macro(cuptiGetTimestamp);                 \
  __macro(cuptiActivityGetNextRecord);        \
  __macro(cuptiGetResultString);              \
  __macro(cuptiActivityGetNumDroppedRecords); \
  __macro(cuptiActivityFlushAll);             \
  __macro(cuptiSubscribe);                    \
  __macro(cuptiUnsubscribe);                  \
  __macro(cuptiEnableCallback);               \
  __macro(cuptiEnableDomain);

CUPTI_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_CUPTI_WRAP);

#undef DECLARE_DYNAMIC_LOAD_CUPTI_WRAP
}  // namespace dynload
}  // namespace platform
}  // namespace paddle

#endif  // PADDLE_WITH_CUPTI
