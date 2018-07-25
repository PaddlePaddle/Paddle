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
#include <dlfcn.h>

#include <mutex>  // NOLINT

#include "paddle/fluid/platform/dynload/dynamic_loader.h"

namespace paddle {
namespace platform {
namespace dynload {
extern std::once_flag curand_dso_flag;
extern void *curand_dso_handle;
#ifdef PADDLE_USE_DSO
#define DECLARE_DYNAMIC_LOAD_CURAND_WRAP(__name)                             \
  struct DynLoad__##__name {                                                 \
    template <typename... Args>                                              \
    curandStatus_t operator()(Args... args) {                                \
      using curandFunc = decltype(&::__name);                                \
      std::call_once(curand_dso_flag, []() {                                 \
        curand_dso_handle = paddle::platform::dynload::GetCurandDsoHandle(); \
      });                                                                    \
      static void *p_##__name = dlsym(curand_dso_handle, #__name);           \
      return reinterpret_cast<curandFunc>(p_##__name)(args...);              \
    }                                                                        \
  };                                                                         \
  extern DynLoad__##__name __name
#else
#define DECLARE_DYNAMIC_LOAD_CURAND_WRAP(__name) \
  struct DynLoad__##__name {                     \
    template <typename... Args>                  \
    curandStatus_t operator()(Args... args) {    \
      return __name(args...);                    \
    }                                            \
  };                                             \
  extern DynLoad__##__name __name
#endif

#define CURAND_RAND_ROUTINE_EACH(__macro)      \
  __macro(curandCreateGenerator);              \
  __macro(curandSetStream);                    \
  __macro(curandSetPseudoRandomGeneratorSeed); \
  __macro(curandGenerate);                     \
  __macro(curandGenerateUniform);              \
  __macro(curandGenerateUniformDouble);        \
  __macro(curandGenerateNormal);               \
  __macro(curandDestroyGenerator);

CURAND_RAND_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_CURAND_WRAP);

}  // namespace dynload

// From Caffe
inline const char *curandGetErrorString(curandStatus_t error) {
  switch (error) {
    case CURAND_STATUS_SUCCESS:
      return "CURAND_STATUS_SUCCESS";
    case CURAND_STATUS_VERSION_MISMATCH:
      return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED:
      return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED:
      return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR:
      return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE:
      return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE:
      return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH:
      return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR:
      return "CURAND_STATUS_INTERNAL_ERROR";
  }
  return "Unknown curand status";
}

#define CURAND_ENFORCE(condition)                           \
  do {                                                      \
    auto status = condition;                                \
    if (status != CURAND_STATUS_SUCCESS) {                  \
      const char *error_string =                            \
          ::paddle::platform::curandGetErrorString(status); \
      VLOG(1) << error_string;                              \
      PADDLE_THROW("cuRAND call failed: %s", error_string); \
    }                                                       \
  } while (false)

}  // namespace platform
}  // namespace paddle
