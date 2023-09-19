/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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
#include "warprnnt/include/rnnt.h"

namespace phi {
namespace dynload {

extern std::once_flag warprnnt_dso_flag;
extern void* warprnnt_dso_handle;

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load warprnnt routine
 * via operator overloading.
 */
#define DYNAMIC_LOAD_WARPRNNT_WRAP(__name)                           \
  struct DynLoad__##__name {                                         \
    template <typename... Args>                                      \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { \
      using warprnntFunc = decltype(&::__name);                      \
      std::call_once(warprnnt_dso_flag, []() {                       \
        warprnnt_dso_handle = phi::dynload::GetWarpRNNTDsoHandle();  \
      });                                                            \
      static void* p_##__name = dlsym(warprnnt_dso_handle, #__name); \
      return reinterpret_cast<warprnntFunc>(p_##__name)(args...);    \
    }                                                                \
  };                                                                 \
  extern DynLoad__##__name __name

#define DECLARE_DYNAMIC_LOAD_WARPRNNT_WRAP(__name) \
  DYNAMIC_LOAD_WARPRNNT_WRAP(__name)

#define WARPRNNT_ROUTINE_EACH(__macro) \
  __macro(get_warprnnt_version);       \
  __macro(rnntGetStatusString);        \
  __macro(compute_rnnt_loss);          \
  __macro(compute_rnnt_loss_fp64);     \
  __macro(get_rnnt_workspace_size);

WARPRNNT_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_WARPRNNT_WRAP);

#undef DYNAMIC_LOAD_WARPRNNT_WRAP

}  // namespace dynload
}  // namespace phi
