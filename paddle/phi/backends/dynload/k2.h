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

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/backends/dynload/port.h"

namespace phi {
namespace dynload {

extern std::once_flag k2_dso_flag;
extern void* k2_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load k2 routine
 * via operator overloading.
 */
#define DYNAMIC_LOAD_K2_WRAP(__name)                                 \
  struct DynLoad__##__name {                                         \
    template <typename... Args>                                      \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { \
      using k2Func = decltype(&::__name);                            \
      std::call_once(k2_dso_flag, []() {                             \
        k2_dso_handle = phi::dynload::GetK2LogDsoHandle();           \
      });                                                            \
      static void* p_##__name = dlsym(k2_dso_handle, #__name);       \
      return reinterpret_cast<k2Func>(p_##__name)(args...);          \
    }                                                                \
  };                                                                 \
  extern DynLoad__##__name __name

#define DECLARE_DYNAMIC_LOAD_K2_WRAP(__name)  // DYNAMIC_LOAD_K2_WRAP(__name)

#define K2_ROUTINE_EACH(__macro) \
  __macro(Logger);               \
  __macro(EnableAbort)

K2_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_K2_WRAP);

#undef DYNAMIC_LOAD_K2_WRAP

// void* GetK2ContextDsoHandle();
// void* GetK2FsaDsoHandle();
// void* GetK2LogDsoHandle();

}  // namespace dynload
}  // namespace phi
