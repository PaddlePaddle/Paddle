/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

#include "flashattn/include/flash_attn.h"
#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/common/port.h"

namespace phi {
namespace dynload {

extern std::once_flag flashattn_dso_flag;
extern void* flashattn_dso_handle;

#define DYNAMIC_LOAD_FLASHATTN_WRAP(__name)                           \
  struct DynLoad__##__name {                                          \
    template <typename... Args>                                       \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) {  \
      using flashattnFunc = decltype(&::__name);                      \
      std::call_once(flashattn_dso_flag, []() {                       \
        flashattn_dso_handle = phi::dynload::GetFlashAttnDsoHandle(); \
      });                                                             \
      static void* p_##__name = dlsym(flashattn_dso_handle, #__name); \
      return reinterpret_cast<flashattnFunc>(p_##__name)(args...);    \
    }                                                                 \
  };                                                                  \
  extern DynLoad__##__name __name

#define DECLARE_DYNAMIC_LOAD_FLASHATTN_WRAP(__name) \
  DYNAMIC_LOAD_FLASHATTN_WRAP(__name)

#ifdef PADDLE_WITH_HIP
#define FLASHATTN_ROUTINE_EACH(__macro) \
  __macro(flash_attn_fwd);              \
  __macro(flash_attn_varlen_fwd);       \
  __macro(flash_attn_bwd);              \
  __macro(flash_attn_varlen_bwd);       \
  __macro(flash_attn_error);
#else
#define FLASHATTN_ROUTINE_EACH(__macro)       \
  __macro(flash_attn_fwd);                    \
  __macro(flash_attn_varlen_fwd);             \
  __macro(flash_attn_bwd);                    \
  __macro(flash_attn_varlen_bwd);             \
  __macro(calc_reduced_attn_scores);          \
  __macro(flash_attn_fwd_with_bias_and_mask); \
  __macro(flash_attn_bwd_with_bias_and_mask); \
  __macro(flash_attn_error);
#endif

FLASHATTN_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_FLASHATTN_WRAP);

#undef DYNAMIC_LOAD_FLASHATTN_WRAP

}  // namespace dynload
}  // namespace phi
