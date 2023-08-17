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

#include <murand.h>

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/backends/dynload/port.h"

namespace phi {
namespace dynload {
extern std::once_flag murand_dso_flag;
extern void *murand_dso_handle;

#define DECLARE_DYNAMIC_LOAD_MURAND_WRAP(__name)                   \
  struct DynLoad__##__name {                                       \
    template <typename... Args>                                    \
    murandStatus_t operator()(Args... args) {                      \
      using murandFunc = decltype(&::__name);                      \
      std::call_once(murand_dso_flag, []() {                       \
        murand_dso_handle = phi::dynload::GetCurandDsoHandle();    \
      });                                                          \
      static void *p_##__name = dlsym(murand_dso_handle, #__name); \
      return reinterpret_cast<murandFunc>(p_##__name)(args...);    \
    }                                                              \
  };                                                               \
  extern DynLoad__##__name __name

#define MURAND_RAND_ROUTINE_EACH(__macro)      \
  __macro(murandCreateGenerator);              \
  __macro(murandSetStream);                    \
  __macro(murandSetPseudoRandomGeneratorSeed); \
  __macro(murandGenerateUniform);              \
  __macro(murandGenerateUniformDouble);        \
  __macro(murandGenerateNormal);               \
  __macro(murandDestroyGenerator);

MURAND_RAND_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_MURAND_WRAP);

}  // namespace dynload
}  // namespace phi
