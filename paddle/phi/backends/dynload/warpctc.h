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
#include "warpctc/include/ctc.h"

namespace phi {
namespace dynload {

extern std::once_flag warpctc_dso_flag;
extern void* warpctc_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load warpctc routine
 * via operator overloading.
 */
#define DYNAMIC_LOAD_WARPCTC_WRAP(__name)                            \
  struct DynLoad__##__name {                                         \
    template <typename... Args>                                      \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { \
      using warpctcFunc = decltype(&::__name);                       \
      std::call_once(warpctc_dso_flag, []() {                        \
        warpctc_dso_handle = phi::dynload::GetWarpCTCDsoHandle();    \
      });                                                            \
      static void* p_##__name = dlsym(warpctc_dso_handle, #__name);  \
      return reinterpret_cast<warpctcFunc>(p_##__name)(args...);     \
    }                                                                \
  };                                                                 \
  extern DynLoad__##__name __name

#define DECLARE_DYNAMIC_LOAD_WARPCTC_WRAP(__name) \
  DYNAMIC_LOAD_WARPCTC_WRAP(__name)

#define WARPCTC_ROUTINE_EACH(__macro) \
  __macro(get_warpctc_version);       \
  __macro(ctcGetStatusString);        \
  __macro(compute_ctc_loss);          \
  __macro(compute_ctc_loss_double);   \
  __macro(get_workspace_size);        \
  __macro(get_workspace_size_double)

WARPCTC_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_WARPCTC_WRAP);

#undef DYNAMIC_LOAD_WARPCTC_WRAP

}  // namespace dynload
}  // namespace phi
