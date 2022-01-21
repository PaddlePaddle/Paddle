/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/pten/backends/dynload/hiprand.h"

namespace paddle {
namespace platform {
namespace dynload {

#define PLATFORM_DECLARE_DYNAMIC_LOAD_CURAND_WRAP(__name)     \
  using DynLoad__##__name = pten::dynload::DynLoad__##__name; \
  extern DynLoad__##__name __name

#define HIPRAND_RAND_ROUTINE_EACH(__macro)      \
  __macro(hiprandCreateGenerator);              \
  __macro(hiprandSetStream);                    \
  __macro(hiprandSetPseudoRandomGeneratorSeed); \
  __macro(hiprandGenerateUniform);              \
  __macro(hiprandGenerateUniformDouble);        \
  __macro(hiprandGenerateNormal);               \
  __macro(hiprandDestroyGenerator);

HIPRAND_RAND_ROUTINE_EACH(PLATFORM_DECLARE_DYNAMIC_LOAD_CURAND_WRAP);

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
