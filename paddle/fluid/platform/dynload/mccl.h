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

#include <mccl.h>

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/mccl.h"

namespace paddle {
namespace platform {
namespace dynload {

#define PLATFORM_DECLARE_DYNAMIC_LOAD_MCCL_WRAP(__name)      \
  using DynLoad__##__name = phi::dynload::DynLoad__##__name; \
  extern DynLoad__##__name __name

MCCL_RAND_ROUTINE_EACH(PLATFORM_DECLARE_DYNAMIC_LOAD_MCCL_WRAP)

#define MCCL_RAND_ROUTINE_EACH_AFTER_2212(__macro) __macro(mcclBroadcast);
MCCL_RAND_ROUTINE_EACH_AFTER_2212(PLATFORM_DECLARE_DYNAMIC_LOAD_MCCL_WRAP)

#define MCCL_RAND_ROUTINE_EACH_AFTER_2304(__macro) __macro(mcclGetVersion);
MCCL_RAND_ROUTINE_EACH_AFTER_2304(PLATFORM_DECLARE_DYNAMIC_LOAD_MCCL_WRAP)

#define MCCL_RAND_ROUTINE_EACH_AFTER_2703(__macro) \
  __macro(mcclSend);                               \
  __macro(mcclRecv);
MCCL_RAND_ROUTINE_EACH_AFTER_2703(PLATFORM_DECLARE_DYNAMIC_LOAD_MCCL_WRAP)


#define MCCL_RAND_ROUTINE_EACH_AFTER_21100(__macro) \
  __macro(mcclRedOpCreatePreMulSum);                \
  __macro(mcclRedOpDestroy);
MCCL_RAND_ROUTINE_EACH_AFTER_21100(PLATFORM_DECLARE_DYNAMIC_LOAD_MCCL_WRAP)

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
