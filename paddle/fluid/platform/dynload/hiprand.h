<<<<<<< HEAD
/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
=======
/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
>>>>>>> 3dfaefa74fbdc4d9d2b95db145e43d0f01cac198

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

#include <hiprand.h>

#include <mutex>  // NOLINT
#include "paddle/fluid/platform/port.h"

#include "paddle/fluid/platform/dynload/dynamic_loader.h"

namespace paddle {
namespace platform {
namespace dynload {
<<<<<<< HEAD
extern std::once_flag hiprand_dso_flag;
extern void *hiprand_dso_handle;

#define DECLARE_DYNAMIC_LOAD_CURAND_WRAP(__name)                              \
  struct DynLoad__##__name {                                                  \
    template <typename... Args>                                               \
    hiprandStatus_t operator()(Args... args) {                                \
      using hiprandFunc = decltype(&::__name);                                \
      std::call_once(hiprand_dso_flag, []() {                                 \
        hiprand_dso_handle = paddle::platform::dynload::GetCurandDsoHandle(); \
      });                                                                     \
      static void *p_##__name = dlsym(hiprand_dso_handle, #__name);           \
      return reinterpret_cast<hiprandFunc>(p_##__name)(args...);              \
    }                                                                         \
  };                                                                          \
  extern DynLoad__##__name __name

#define HIPRAND_RAND_ROUTINE_EACH(__macro)      \
=======
extern std::once_flag curand_dso_flag;
extern void *curand_dso_handle;

#define DECLARE_DYNAMIC_LOAD_CURAND_WRAP(__name)                             \
  struct DynLoad__##__name {                                                 \
    template <typename... Args>                                              \
    hiprandStatus_t operator()(Args... args) {                                \
      using curandFunc = decltype(&::__name);                                \
      std::call_once(curand_dso_flag, []() {                                 \
        curand_dso_handle = paddle::platform::dynload::GetCurandDsoHandle(); \
      });                                                                    \
      static void *p_##__name = dlsym(curand_dso_handle, #__name);           \
      return reinterpret_cast<curandFunc>(p_##__name)(args...);              \
    }                                                                        \
  };                                                                         \
  extern DynLoad__##__name __name

#define CURAND_RAND_ROUTINE_EACH(__macro)      \
>>>>>>> 3dfaefa74fbdc4d9d2b95db145e43d0f01cac198
  __macro(hiprandCreateGenerator);              \
  __macro(hiprandSetStream);                    \
  __macro(hiprandSetPseudoRandomGeneratorSeed); \
  __macro(hiprandGenerateUniform);              \
  __macro(hiprandGenerateUniformDouble);        \
  __macro(hiprandGenerateNormal);               \
  __macro(hiprandDestroyGenerator);

<<<<<<< HEAD
HIPRAND_RAND_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_CURAND_WRAP);
=======
CURAND_RAND_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_CURAND_WRAP);
>>>>>>> 3dfaefa74fbdc4d9d2b95db145e43d0f01cac198

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
