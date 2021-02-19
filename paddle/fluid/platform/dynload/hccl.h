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

#include <hccl/hccl.h>
#include <hccl/hccl_types.h>
#include <mutex>  // NOLINT

#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/port.h"

HcclResult HcomCreateGroup(const char *group, uint32_t rankNum, uint32_t *rankIds);
HcclResult HcomSend(const char *tag, void *inputPtr, uint64_t count, HcclDataType dataType,
    uint32_t destRank, uint32_t srTag, const char *group, void* stream);
HcclResult HcomReceive(const char *tag, void *outputPtr, uint64_t count, HcclDataType dataType,
    uint32_t srcRank, uint32_t srTag, const char *group, void* stream);

namespace paddle {
namespace platform {
namespace dynload {

extern std::once_flag hccl_dso_flag;
extern void* hccl_dso_handle;

#define DECLARE_DYNAMIC_LOAD_HCCL_WRAP(__name)                           \
  struct DynLoad__##__name {                                             \
    template <typename... Args>                                          \
    auto operator()(Args... args) -> decltype(__name(args...)) {         \
      using HCCL_func = decltype(&::__name);                             \
      std::call_once(hccl_dso_flag, []() {                               \
        hccl_dso_handle = paddle::platform::dynload::GetHCCLDsoHandle(); \
      });                                                                \
      static void* p_##__name = dlsym(hccl_dso_handle, #__name);         \
      return reinterpret_cast<HCCL_func>(p_##__name)(args...);           \
    }                                                                    \
  };                                                                     \
  extern DynLoad__##__name __name

#define HCCL_RAND_ROUTINE_EACH(__macro) \
  __macro(HcclCommInitClusterInfo);     \
  __macro(HcclGetRootInfo);             \
  __macro(HcclCommInitRootInfo);        \
  __macro(HcclAllReduce);               \
  __macro(HcclBroadcast);               \
  __macro(HcomCreateGroup);             \
  __macro(HcomSend);                    \
  __macro(HcomReceive);                 \
  __macro(HcclAllGather);               \
  __macro(HcclReduceScatter);           \
  __macro(HcclCommDestroy); 

HCCL_RAND_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_HCCL_WRAP)

#if HCCL_VERSION_CODE >= 2212
#define HCCL_RAND_ROUTINE_EACH_AFTER_2212(__macro) __macro(HCCLBroadcast);
HCCL_RAND_ROUTINE_EACH_AFTER_2212(DECLARE_DYNAMIC_LOAD_HCCL_WRAP)
#endif

#if HCCL_VERSION_CODE >= 2703
#define HCCL_RAND_ROUTINE_EACH_AFTER_2703(__macro) \
  __macro(HCCLSend);                               \
  __macro(HCCLRecv);
HCCL_RAND_ROUTINE_EACH_AFTER_2703(DECLARE_DYNAMIC_LOAD_HCCL_WRAP)
#endif

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
