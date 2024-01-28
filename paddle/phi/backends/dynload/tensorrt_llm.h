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

#include <tllmPlugin.h>
#include <mutex>  // NOLINT

#include "paddle/phi/backends/dynload/dynamic_loader.h"
#include "paddle/phi/core/enforce.h"

namespace phi {
namespace dynload {

extern std::once_flag tensorrt_llm_dso_flag;
extern void* tensorrt_llm_dso_handle;

extern std::once_flag tensorrt_llm_plugin_dso_flag;
extern void* tensorrt_llm_plugin_dso_handle;

#define DECLARE_DYNAMIC_LOAD_TENSORRTLLM_POINTER_WRAP(__name)              \
  struct DynLoad__##__name {                                               \
    template <typename... Args>                                            \
    void* operator()(Args... args) {                                       \
      std::call_once(tensorrt_llm_dso_flag, []() {                         \
        tensorrt_llm_dso_handle = phi::dynload::GetTensorRtLLMDsoHandle(); \
      });                                                                  \
      static void* p_##__name = dlsym(tensorrt_llm_dso_handle, #__name);   \
      if (p_##__name == nullptr) {                                         \
        return nullptr;                                                    \
      }                                                                    \
      using tensorrt_llm_func = decltype(&::__name);                       \
      auto ret = reinterpret_cast<tensorrt_llm_func>(p_##__name)(args...); \
      return static_cast<void*>(ret);                                      \
    }                                                                      \
  };                                                                       \
  extern DynLoad__##__name __name

#define DECLARE_DYNAMIC_LOAD_TENSORRTLLM_PLUGIN_WRAP(__name)                  \
  struct DynLoad__##__name {                                                  \
    template <typename... Args>                                               \
    auto operator()(Args... args) -> DECLARE_TYPE(__name, args...) {          \
      std::call_once(tensorrt_llm_plugin_dso_flag, []() {                     \
        tensorrt_llm_plugin_dso_handle =                                      \
            phi::dynload::GetTensorRtLLMPluginDsoHandle();                    \
      });                                                                     \
      static void* p_##__name =                                               \
          dlsym(tensorrt_llm_plugin_dso_handle, #__name);                     \
      PADDLE_ENFORCE_NOT_NULL(                                                \
          p_##__name,                                                         \
          phi::errors::Unavailable("Load tensorrt_llm plugin %s failed",      \
                                   #__name));                                 \
      using tensorrt_llm_plugin_func = decltype(&::__name);                   \
      return reinterpret_cast<tensorrt_llm_plugin_func>(p_##__name)(args...); \
    }                                                                         \
  };                                                                          \
  extern DynLoad__##__name __name

#define TENSORRTLLM_RAND_ROUTINE_EACH_POINTER(__macro)

#define TENSORRTLLM_PLUGIN_RAND_ROUTINE_EACH(__macro) \
  __macro(initTrtLlmPlugins);

TENSORRTLLM_RAND_ROUTINE_EACH_POINTER(
    DECLARE_DYNAMIC_LOAD_TENSORRTLLM_POINTER_WRAP)
TENSORRTLLM_PLUGIN_RAND_ROUTINE_EACH(
    DECLARE_DYNAMIC_LOAD_TENSORRTLLM_PLUGIN_WRAP)

}  // namespace dynload
}  // namespace phi
