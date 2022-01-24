/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/platform/device/xpu/xpu_header.h"

#include "paddle/pten/backends/xpu/enforce_xpu.h"

namespace paddle {
namespace platform {

// Note: XPU runtime api return int, not XPUError_t
inline const char* xpuGetErrorString(int stat) {
  return pten::backends::xpu::xpuGetErrorString(stat);
}

inline const char* bkclGetErrorString(BKCLResult_t stat) {
  return pten::backends::xpu::bkclGetErrorString(stat);
}

inline const char* xdnnGetErrorString(int stat) {
  return pten::backends::xpu::xdnnGetErrorString(stat);
}

inline std::string build_xpu_error_msg(int stat) {
  return pten::backends::xpu::build_xpu_error_msg(stat);
}

inline std::string build_xpu_error_msg(BKCLResult_t stat) {
  return pten::backends::xpu::build_xpu_error_msg(stat);
}

inline std::string build_xpu_xdnn_error_msg(int stat, std::string msg) {
  return pten::backends::xpu::build_xpu_xdnn_error_msg(stat, msg);
}

namespace details {

template <typename T>
struct ExternalApiType {};

#define DEFINE_EXTERNAL_API_TYPE(type, success_value) \
  template <>                                         \
  struct ExternalApiType<type> {                      \
    using Type = type;                                \
    static constexpr Type kSuccess = success_value;   \
  }

DEFINE_EXTERNAL_API_TYPE(int, XPU_SUCCESS);
DEFINE_EXTERNAL_API_TYPE(BKCLResult_t, BKCL_SUCCESS);

#undef DEFINE_EXTERNAL_API_TYPE

}  // namespace details

#define PADDLE_ENFORCE_XPU_SUCCESS(COND)                      \
  do {                                                        \
    auto __cond__ = (COND);                                   \
    using __XPU_STATUS_TYPE__ = decltype(__cond__);           \
    constexpr auto __success_type__ =                         \
        ::paddle::platform::details::ExternalApiType<         \
            __XPU_STATUS_TYPE__>::kSuccess;                   \
    if (UNLIKELY(__cond__ != __success_type__)) {             \
      auto __summary__ = paddle::platform::errors::External(  \
          ::paddle::platform::build_xpu_error_msg(__cond__)); \
      __THROW_ERROR_INTERNAL__(__summary__);                  \
    }                                                         \
  } while (0)

#define PADDLE_ENFORCE_XDNN_NOT_NULL(ptr)                                    \
  do {                                                                       \
    PADDLE_ENFORCE_NOT_NULL(                                                 \
        ptr, ::paddle::platform::errors::Fatal("XPU memory is not enough")); \
  } while (0)

#define PADDLE_ENFORCE_XDNN_SUCCESS(COND, MSG)                          \
  do {                                                                  \
    auto __cond__ = (COND);                                             \
    if (UNLIKELY(__cond__ != xpu::Error_t::SUCCESS)) {                  \
      auto __summary__ = paddle::platform::errors::External(            \
          ::paddle::platform::build_xpu_xdnn_error_msg(__cond__, MSG)); \
      __THROW_ERROR_INTERNAL__(__summary__);                            \
    }                                                                   \
  } while (0)

}  // namespace platform
}  // namespace paddle
