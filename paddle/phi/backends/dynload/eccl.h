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

#include <eccl.h>
#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/backends/dynload/port.h"

namespace phi {
namespace dynload {

extern std::once_flag eccl_dso_flag;
extern void* eccl_dso_handle;

#define DECLARE_DYNAMIC_LOAD_ECCL_WRAP(__name)                   \
  struct DynLoad__##__name {                                     \
    template <typename... Args>                                  \
    auto operator()(Args... args) -> decltype(__name(args...)) { \
      using eccl_func = decltype(&::__name);                     \
      std::call_once(eccl_dso_flag, []() {                       \
        eccl_dso_handle = phi::dynload::GetECCLDsoHandle();      \
      });                                                        \
      static void* p_##__name = dlsym(eccl_dso_handle, #__name); \
      return reinterpret_cast<eccl_func>(p_##__name)(args...);   \
    }                                                            \
  };                                                             \
  extern DynLoad__##__name __name

#define ECCL_RAND_ROUTINE_EACH(__macro) \
  __macro(eccl_init_comm_global);       \
  __macro(eccl_gen_unique_id);          \
  __macro(ecclCommInitRank);            \
  __macro(eccl_destroy_comm_global);    \
  __macro(eccl_reduce);                 \
  __macro(eccl_broadcast);              \
  __macro(eccl_all_reduce);             \
  __macro(eccl_reduce_scatter);         \
  __macro(eccl_all_gather);             \
  __macro(eccl_send);                   \
  __macro(eccl_recv);                   \
  __macro(eccl_group_begin);            \
  __macro(eccl_group_end);

ECCL_RAND_ROUTINE_EACH(DECLARE_DYNAMIC_LOAD_ECCL_WRAP)

}  // namespace dynload
}  // namespace phi
