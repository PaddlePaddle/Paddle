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

#include <mutex>  // NOLINT

#include "paddle/pten/backends/dynload/warpctc.h"

namespace paddle {
namespace platform {
namespace dynload {

/**
 * The following macro definition can generate structs
 * (for each function) to dynamic load warpctc routine
 * via operator overloading.
 */
#define DYNAMIC_LOAD_WARPCTC_WRAP(__name)                     \
  using DynLoad__##__name = pten::dynload::DynLoad__##__name; \
  extern DynLoad__##__name __name

#define DECLARE_DYNAMIC_LOAD_WARPCTC_WRAP(__name) \
  DYNAMIC_LOAD_WARPCTC_WRAP(__name)

#define WARPCTC_ROUTINE_EACH(__macro) \
  __macro(get_warpctc_version);       \
  __macro(ctcGetStatusString);        \
  __macro(compute_ctc_loss);          \
  __macro(compute_ctc_loss_double);   \
  __macro(get_workspace_size);        \
  __macro(get_workspace_size_double)

WARPCTC_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_WARPCTC_WRAP);

#undef DYNAMIC_LOAD_WARPCTC_WRAP

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
