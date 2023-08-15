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
#ifdef PADDLE_WITH_MUSA
#include <mudnn.h>

#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/backends/dynload/port.h"

namespace phi {
namespace dynload {

extern std::once_flag mudnn_dso_flag;
extern void* mudnn_dso_handle;
extern bool HasCUDNN();

extern void EnforceCUDNNLoaded(const char* fn_name);
#define DECLARE_DYNAMIC_LOAD_CUDNN_WRAP(__name)                      \
  struct DynLoad__##__name {                                         \
    template <typename... Args>                                      \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) { \
      using mudnn_func = decltype(&::__name);                        \
      std::call_once(mudnn_dso_flag, []() {                          \
        mudnn_dso_handle = phi::dynload::GetCUDNNDsoHandle();        \
      });                                                            \
      EnforceCUDNNLoaded(#__name);                                   \
      static void* p_##__name = dlsym(mudnn_dso_handle, #__name);    \
      return reinterpret_cast<mudnn_func>(p_##__name)(args...);      \
    }                                                                \
  };                                                                 \
  extern struct DynLoad__##__name __name

}  // namespace dynload
}  // namespace phi
#endif
