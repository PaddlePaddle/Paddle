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

#include <rccl.h>

#include <mutex>  // NOLINT
#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/port.h"

namespace paddle {
namespace platform {
namespace dynload {

<<<<<<< HEAD
extern std::once_flag rccl_dso_flag;
extern void* rccl_dso_handle;

#define DECLARE_DYNAMIC_LOAD_RCCL_WRAP(__name)                           \
=======
extern std::once_flag nccl_dso_flag;
extern void* nccl_dso_handle;

#define DECLARE_DYNAMIC_LOAD_NCCL_WRAP(__name)                           \
>>>>>>> 3dfaefa74fbdc4d9d2b95db145e43d0f01cac198
  struct DynLoad__##__name {                                             \
    template <typename... Args>                                          \
    auto operator()(Args... args) -> decltype(__name(args...)) {         \
      using nccl_func = decltype(&::__name);                             \
<<<<<<< HEAD
      std::call_once(rccl_dso_flag, []() {                               \
        rccl_dso_handle = paddle::platform::dynload::GetNCCLDsoHandle(); \
      });                                                                \
      static void* p_##__name = dlsym(rccl_dso_handle, #__name);         \
=======
      std::call_once(nccl_dso_flag, []() {                               \
        nccl_dso_handle = paddle::platform::dynload::GetNCCLDsoHandle(); \
      });                                                                \
      static void* p_##__name = dlsym(nccl_dso_handle, #__name);         \
>>>>>>> 3dfaefa74fbdc4d9d2b95db145e43d0f01cac198
      return reinterpret_cast<nccl_func>(p_##__name)(args...);           \
    }                                                                    \
  };                                                                     \
  extern DynLoad__##__name __name

<<<<<<< HEAD
#define RCCL_RAND_ROUTINE_EACH(__macro) \
=======
#define NCCL_RAND_ROUTINE_EACH(__macro) \
>>>>>>> 3dfaefa74fbdc4d9d2b95db145e43d0f01cac198
  __macro(ncclCommInitAll);             \
  __macro(ncclGetUniqueId);             \
  __macro(ncclCommInitRank);            \
  __macro(ncclCommDestroy);             \
  __macro(ncclCommCount);               \
  __macro(ncclCommCuDevice);            \
  __macro(ncclCommUserRank);            \
  __macro(ncclAllReduce);               \
  __macro(ncclBcast);                   \
  __macro(ncclAllGather);               \
  __macro(ncclGroupStart);              \
  __macro(ncclGroupEnd);                \
  __macro(ncclReduce);                  \
  __macro(ncclReduceScatter);           \
  __macro(ncclGetErrorString);

<<<<<<< HEAD
RCCL_RAND_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_RCCL_WRAP)
=======
NCCL_RAND_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_NCCL_WRAP)
>>>>>>> 3dfaefa74fbdc4d9d2b95db145e43d0f01cac198

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
