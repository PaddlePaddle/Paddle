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

#include <mublas.h>
#include <musa.h>

#include <mutex>  // NOLINT
#include <type_traits>

#include "paddle/phi/backends/dynload/mublas.h"

namespace paddle {
namespace platform {
namespace dynload {

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load mublas routine
 * via operator overloading.
 *
 * note: default dynamic linked libs
 */
#define PLATFORM_DECLARE_DYNAMIC_LOAD_MUBLAS_WRAP(__name)    \
  using DynLoad__##__name = phi::dynload::DynLoad__##__name; \
  extern DynLoad__##__name __name


MUBLAS_BLAS_ROUTINE_EACH(PLATFORM_DECLARE_DYNAMIC_LOAD_MUBLAS_WRAP)


MUBLAS_BLAS_ROUTINE_EACH_R2(PLATFORM_DECLARE_DYNAMIC_LOAD_MUBLAS_WRAP)


MUBLAS_BLAS_ROUTINE_EACH_R3(PLATFORM_DECLARE_DYNAMIC_LOAD_MUBLAS_WRAP)


MUBLAS_BLAS_ROUTINE_EACH_R4(PLATFORM_DECLARE_DYNAMIC_LOAD_MUBLAS_WRAP)

#undef PLATFORM_DECLARE_DYNAMIC_LOAD_MUBLAS_WRAP
}  // namespace dynload
}  // namespace platform
}  // namespace paddle
