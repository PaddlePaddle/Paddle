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

#ifdef PADDLE_WITH_XPTI

#include <xpu/xpti.h>

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/xpti.h"

namespace paddle {
namespace platform {
namespace dynload {

#define DECLARE_DYNAMIC_LOAD_XPTI_WRAP(__name)               \
  using DynLoad__##__name = phi::dynload::DynLoad__##__name; \
  extern DynLoad__##__name __name

#define XPTI_RAND_ROUTINE_EACH(__macro) \
  __macro(xptiActivityEnable);          \
  __macro(xptiActivityDisable);         \
  __macro(xptiStartTracing);            \
  __macro(xptiStopTracing);             \
  __macro(xptiActivityFlushAll);        \
  __macro(xptiActivityGetNextRecord);

XPTI_RAND_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_XPTI_WRAP);

#undef DECLARE_DYNAMIC_LOAD_XPTI_WRAP
}  // namespace dynload
}  // namespace platform
}  // namespace paddle

#endif  // PADDLE_WITH_XPTI
