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

#ifdef PADDLE_WITH_ECCL

#include <eccl/eccl.h>
#include <eccl/eccl_types.h>
#include <mutex>  // NOLINT

#include "paddle/fluid/platform/dynload/dynamic_loader.h"
#include "paddle/fluid/platform/port.h"

#define HCOM_GROUP_PREFIX "HCOM_GROUP_"

namespace paddle {
namespace platform {
namespace dynload {

extern std::once_flag eccl_dso_flag;
extern void* eccl_dso_handle;

#define DECLARE_DYNAMIC_LOAD_ECCL_WRAP(__name)                           \
  struct DynLoad__##__name {                                             \
    template <typename... Args>                                          \
    auto operator()(Args... args) -> decltype(__name(args...)) {         \
      using eccl_func = decltype(&::__name);                             \
      std::call_once(eccl_dso_flag, []() {                               \
        eccl_dso_handle = paddle::platform::dynload::GetECCLDsoHandle(); \
      });                                                                \
      static void* p_##__name = dlsym(eccl_dso_handle, #__name);         \
      return reinterpret_cast<eccl_func>(p_##__name)(args...);           \
    }                                                                    \
  };                                                                     \
  extern DynLoad__##__name __name

#define ECCL_RAND_ROUTINE_EACH(__macro) \
  __macro(eccl_gen_unique_id);          \
  __macro(eccl_init_comm_global);       \
  __macro(eccl_destroy_comm_global);    \
  __macro(h_eccl_reduce_scatter);       \
  __macro(h_eccl_all_reduce);           \
  __macro(h_eccl_broadcast);            \
  __macro(h_eccl_all_gather);           \
  __macro(h_eccl_reduce);               \
  __macro(eccl_sync_stream);

ECCL_RAND_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_ECCL_WRAP)

}  // namespace dynload
}  // namespace platform
}  // namespace paddle
#endif
