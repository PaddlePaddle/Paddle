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

#ifdef PADDLE_WITH_XPU
#include <cuda.h>
#include <cuda_runtime.h>
#include <xpu/xpuml.h>
#endif

#include "paddle/phi/backends/xpu/xpu_header.h"
#include "paddle/phi/core/enforce.h"
#include "xre/cuda_runtime_api.h"

namespace phi {
namespace backends {
namespace xpu {

std::string get_xpu_error_msg(int error_code);
#ifdef PADDLE_WITH_XPU_BKCL
std::string get_bkcl_error_msg(BKCLResult_t stat);
#endif
std::string get_xdnn_error_msg(int error_code, std::string msg);
std::string get_xblas_error_msg(int error_code, std::string msg);

namespace details {
template <typename T>
struct ExternalApiType {};

// Macro to define the external API type specialization
#define DEFINE_EXTERNAL_API_TYPE(type, success_value) \
  template <>                                         \
  struct ExternalApiType<type> {                      \
    using Type = type;                                \
    static constexpr Type kSuccess = success_value;   \
  }

// Existing specializations for XPU api
DEFINE_EXTERNAL_API_TYPE(int, XPU_SUCCESS);
// Added specialization for cudaError_t to support CUDA api
DEFINE_EXTERNAL_API_TYPE(cudaError_t, cudaSuccess);
#undef DEFINE_EXTERNAL_API_TYPE
}  // namespace details

// return code type int for xpu api, type cudaError_t for cuda api
#define PADDLE_ENFORCE_XPU_SUCCESS(COND)                      \
  do {                                                        \
    auto __cond__ = (COND);                                   \
    using __XPU_STATUS_TYPE__ = decltype(__cond__);           \
    constexpr auto __success_type__ =                         \
        ::phi::backends::xpu::details::ExternalApiType<       \
            __XPU_STATUS_TYPE__>::kSuccess;                   \
    if (UNLIKELY(__cond__ != __success_type__)) {             \
      std::string error_msg =                                 \
          ::phi::backends::xpu::get_xpu_error_msg(__cond__);  \
      auto __summary__ = common::errors::External(error_msg); \
      __THROW_ERROR_INTERNAL__(__summary__);                  \
    }                                                         \
  } while (0)

// return type BKCLResult_t
#ifdef PADDLE_WITH_XPU_BKCL
#define PADDLE_ENFORCE_BKCL_SUCCESS(COND)                     \
  do {                                                        \
    auto __cond__ = (COND);                                   \
    if (UNLIKELY(__cond__ != BKCLResult_t::BKCL_SUCCESS)) {   \
      std::string error_msg =                                 \
          ::phi::backends::xpu::get_bkcl_error_msg(__cond__); \
      auto __summary__ = common::errors::External(error_msg); \
      __THROW_ERROR_INTERNAL__(__summary__);                  \
    }                                                         \
  } while (0)
#endif

#define PADDLE_ENFORCE_XDNN_SUCCESS(COND, MSG)                     \
  do {                                                             \
    auto __cond__ = (COND);                                        \
    if (UNLIKELY(__cond__ != baidu::xpu::api::Error_t::SUCCESS)) { \
      std::string error_msg =                                      \
          ::phi::backends::xpu::get_xdnn_error_msg(__cond__, MSG); \
      auto __summary__ = common::errors::External(error_msg);      \
      __THROW_ERROR_INTERNAL__(__summary__);                       \
    }                                                              \
  } while (0)

// 由于xblas的错误类型定义源文件(xblas_api.h)不能在.h中直接引用
// 因此使用error_msg是否为空来判断是否发生错误，返回值判断逻辑在.cc中实现
#define PADDLE_ENFORCE_XBLAS_SUCCESS(COND, MSG)                   \
  do {                                                            \
    auto __cond__ = (COND);                                       \
    std::string error_msg =                                       \
        ::phi::backends::xpu::get_xblas_error_msg(__cond__, MSG); \
    if (error_msg != "") {                                        \
      auto __summary__ = common::errors::External(error_msg);     \
      __THROW_ERROR_INTERNAL__(__summary__);                      \
    }                                                             \
  } while (0)

#define PADDLE_ENFORCE_XDNN_NOT_NULL(ptr)                               \
  do {                                                                  \
    if (UNLIKELY(ptr == nullptr)) {                                     \
      std::string error_msg = ::phi::backends::xpu::get_xdnn_error_msg( \
          baidu::xpu::api::Error_t::NO_ENOUGH_WORKSPACE, "XPU Alloc");  \
      auto __summary__ = common::errors::External(error_msg);           \
      __THROW_ERROR_INTERNAL__(__summary__);                            \
    }                                                                   \
  } while (0)

#define PADDLE_ENFORCE_XRE_SUCCESS(COND)                            \
  do {                                                              \
    auto __cond__ = (COND);                                         \
    auto xre_msg = xpu_strerror(__cond__);                          \
    if (UNLIKELY(__cond__ != XPU_SUCCESS)) {                        \
      auto __summary__ =                                            \
          common::errors::External("XPU Runtime Error: ", xre_msg); \
      __THROW_ERROR_INTERNAL__(__summary__);                        \
    }                                                               \
  } while (0)

// TODO(lijin23): support fine-grained error msg.
#define PADDLE_ENFORCE_XPUML_SUCCESS(COND)                              \
  do {                                                                  \
    auto __cond__ = (COND);                                             \
    PADDLE_ENFORCE_EQ(                                                  \
        __cond__,                                                       \
        XPUML_SUCCESS,                                                  \
        common::errors::Fatal("XPUML Error, error_code=%d", __cond__)); \
  } while (0)

}  // namespace xpu
}  // namespace backends
}  // namespace phi
