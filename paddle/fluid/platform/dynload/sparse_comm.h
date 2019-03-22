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
#include "dgc/dgc.h"

#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {
namespace dynload {

extern std::once_flag sparse_comm_dso_flag;
extern void* sparse_comm_dso_handle;

#define DECLARE_DYNAMIC_LOAD_SPARSE_COMM_WRAP(__name)                         \
  struct DynLoad__##__name {                                                  \
    template <typename... Args>                                               \
    auto operator()(Args... args)                                             \
        -> decltype(paddle::communication::dgc::__name(args...)) {            \
      using sparse_comm_func = decltype(&paddle::communication::dgc::__name); \
      std::call_once(sparse_comm_dso_flag, []() {                             \
        sparse_comm_dso_handle =                                              \
            paddle::platform::dynload::GetSparseCommDsoHandle();              \
      });                                                                     \
      static void* p_##__name = dlsym(sparse_comm_dso_handle, #__name);       \
      return reinterpret_cast<sparse_comm_func>(p_##__name)(args...);         \
    }                                                                         \
  };                                                                          \
  extern DynLoad__##__name __name

#define SPARSE_COMM_RAND_ROUTINE_EACH(__macro) \
  __macro(k_select);                           \
  __macro(get_buffer_size);                    \
  __macro(sparseAllGReduce);

SPARSE_COMM_RAND_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_SPARSE_COMM_WRAP)

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
