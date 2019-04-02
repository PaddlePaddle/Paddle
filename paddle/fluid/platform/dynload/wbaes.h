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

#ifdef PADDLE_WITH_WBAES

#include <WBAESLib.h>
#include <mutex>  // NOLINT

#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/port.h"

namespace paddle {
namespace platform {
namespace dynload {

extern std::once_flag wbaes_dso_flag;
extern void *wbaes_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load wbaes routine
 * via operator overloading.
 */

#define DYNAMIC_LOAD_WBAES_WRAP(__name)                                    \
  struct DynLoad__##__name {                                               \
    template <typename... Args>                                            \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) {       \
      using wbaesFunc = decltype(&::__name);                               \
      std::call_once(wbaes_dso_flag, []() {                                \
        wbaes_dso_handle = paddle::platform::dynload::GetWBAESDsoHandle(); \
      });                                                                  \
      static void *p_##__name = dlsym(wbaes_dso_handle, #__name);          \
      return reinterpret_cast<wbaesFunc>(p_##__name)(args...);             \
    }                                                                      \
  };                                                                       \
  extern DynLoad__##__name __name

#define DECLARE_DYNAMIC_LOAD_WBAES_WRAP(__name) DYNAMIC_LOAD_WBAES_WRAP(__name)

#define WBAES_ROUTINE_EACH(__macro) __macro(GSECF);

WBAES_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_WBAES_WRAP);

#undef DYNAMIC_LOAD_WBAES_WRAP

}  // namespace dynload
}  // namespace platform
}  // namespace paddle

#endif
