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

#include <curand.h>
#include "paddle/platform/dynamic_loader.h"

namespace paddle {
namespace platform {
namespace dynload {
std::once_flag curand_dso_flag;
void *curand_dso_handle = nullptr;
#ifdef PADDLE_USE_DSO
#define DYNAMIC_LOAD_CURAND_WRAP(__name)                            \
  struct DynLoad__##__name {                                        \
    template <typename... Args>                                     \
    curandStatus_t operator()(Args... args) {                       \
      typedef curandStatus_t (*curandFunc)(Args...);                \
      std::call_once(curand_dso_flag,                               \
                     paddle::platform::dynload::GetCurandDsoHandle, \
                     &curand_dso_handle);                           \
      void *p_##__name = dlsym(curand_dso_handle, #__name);         \
      return reinterpret_cast<curandFunc>(p_##__name)(args...);     \
    }                                                               \
  } __name; /* struct DynLoad__##__name */
#else
#define DYNAMIC_LOAD_CURAND_WRAP(__name)      \
  struct DynLoad__##__name {                  \
    template <typename... Args>               \
    curandStatus_t operator()(Args... args) { \
      return __name(args...);                 \
    }                                         \
  } __name; /* struct DynLoad__##__name */
#endif

/* include all needed curand functions in HPPL */
// clang-format off
#define CURAND_RAND_ROUTINE_EACH(__macro)    \
  __macro(curandCreateGenerator)             \
  __macro(curandSetStream)                   \
  __macro(curandSetPseudoRandomGeneratorSeed)\
  __macro(curandGenerateUniform)             \
  __macro(curandGenerateUniformDouble)       \
  __macro(curandDestroyGenerator)
// clang-format on

CURAND_RAND_ROUTINE_EACH(DYNAMIC_LOAD_CURAND_WRAP)

#undef CURAND_RAND_ROUTINE_EACH
#undef DYNAMIC_LOAD_CURAND_WRAP
}  // namespace dynload
}  // namespace platform
}  // namespace paddle
