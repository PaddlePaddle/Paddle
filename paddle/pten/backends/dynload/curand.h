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

#include <curand.h>
#include <mutex>  // NOLINT

#include "paddle/pten/backends/dynload/dynamic_loader.h"
#include "paddle/pten/backends/dynload/port.h"

namespace pten {
namespace dynload {
extern std::once_flag curand_dso_flag;
extern void *curand_dso_handle;

#define DECLARE_DYNAMIC_LOAD_CURAND_WRAP(__name)                   \
  struct DynLoad__##__name {                                       \
    template <typename... Args>                                    \
    curandStatus_t operator()(Args... args) {                      \
      using curandFunc = decltype(&::__name);                      \
      std::call_once(curand_dso_flag, []() {                       \
        curand_dso_handle = pten::dynload::GetCurandDsoHandle();   \
      });                                                          \
      static void *p_##__name = dlsym(curand_dso_handle, #__name); \
      return reinterpret_cast<curandFunc>(p_##__name)(args...);    \
    }                                                              \
  };                                                               \
  extern DynLoad__##__name __name

#define CURAND_RAND_ROUTINE_EACH(__macro)      \
  __macro(curandCreateGenerator);              \
  __macro(curandSetStream);                    \
  __macro(curandSetPseudoRandomGeneratorSeed); \
  __macro(curandGenerateUniform);              \
  __macro(curandGenerateUniformDouble);        \
  __macro(curandGenerateNormal);               \
  __macro(curandDestroyGenerator);

CURAND_RAND_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_CURAND_WRAP);

}  // namespace dynload
}  // namespace pten
