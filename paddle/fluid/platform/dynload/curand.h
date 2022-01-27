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

#include "paddle/pten/backends/dynload/curand.h"

namespace paddle {
namespace platform {
namespace dynload {

#define PLATFORM_DECLARE_DYNAMIC_LOAD_CURAND_WRAP(__name)     \
  using DynLoad__##__name = pten::dynload::DynLoad__##__name; \
  extern DynLoad__##__name __name

#define CURAND_RAND_ROUTINE_EACH(__macro)      \
  __macro(curandCreateGenerator);              \
  __macro(curandSetStream);                    \
  __macro(curandSetPseudoRandomGeneratorSeed); \
  __macro(curandGenerateUniform);              \
  __macro(curandGenerateUniformDouble);        \
  __macro(curandGenerateNormal);               \
  __macro(curandDestroyGenerator);

CURAND_RAND_ROUTINE_EACH(PLATFORM_DECLARE_DYNAMIC_LOAD_CURAND_WRAP);

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
