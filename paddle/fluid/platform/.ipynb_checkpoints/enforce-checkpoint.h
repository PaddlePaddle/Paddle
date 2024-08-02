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

#ifdef __GNUC__
#include <cxxabi.h>  // for __cxa_demangle
#endif               // __GNUC__

#if !defined(_WIN32)
#include <dlfcn.h>   // dladdr
#include <unistd.h>  // sleep, usleep
#else                // _WIN32
#ifndef NOMINMAX
#define NOMINMAX  // msvc max/min macro conflict with std::min/max
#endif
#include <windows.h>  // GetModuleFileName, Sleep
#endif

#ifdef PADDLE_WITH_CUDA
#include <cublas_v2.h>
#include <cudnn.h>
#include <cufft.h>
#include <curand.h>
#include <cusparse.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#endif  // PADDLE_WITH_CUDA

#ifdef PADDLE_WITH_HIP
#include <hiprand/hiprand.h>
#include <miopen/miopen.h>
#include <rocblas/rocblas.h>
#include <thrust/system/hip/error.h>
#include <thrust/system_error.h>  // NOLINT
#endif

#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#if !defined(_WIN32) && !defined(PADDLE_WITH_MUSL)
#include <execinfo.h>
#endif

#define GLOG_NO_ABBREVIATED_SEVERITIES  // msvc conflict logging with windows.h
#include "glog/logging.h"
#include "paddle/common/errors.h"
#include "paddle/common/flags.h"
#include "paddle/common/macros.h"

#include "paddle/phi/common/port.h"
#include "paddle/utils/string/printf.h"
#include "paddle/utils/string/to_string.h"

#ifdef PADDLE_WITH_CUDA
#include "paddle/phi/backends/dynload/cublas.h"
#include "paddle/phi/backends/dynload/cudnn.h"
#include "paddle/phi/backends/dynload/curand.h"
#include "paddle/phi/backends/dynload/cusolver.h"
#if !defined(__APPLE__) && defined(PADDLE_WITH_NCCL)
#include <error.h>

#include "paddle/phi/backends/dynload/nccl.h"
#endif  // __APPLE__
#endif  // PADDLE_WITH_CUDA

#ifdef PADDLE_WITH_HIP
#include "paddle/phi/backends/dynload/hipfft.h"
#include "paddle/phi/backends/dynload/hiprand.h"
#include "paddle/phi/backends/dynload/miopen.h"
#include "paddle/phi/backends/dynload/rocblas.h"
#if !defined(__APPLE__) && defined(PADDLE_WITH_RCCL)
#include <error.h>  // NOLINT

#include "paddle/phi/backends/dynload/rccl.h"
#endif  // __APPLE__
#endif  // PADDLE_WITH_HIP

// Note: these headers for simplify demangle type string
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/imperative/type_defs.h"
#include "paddle/phi/core/enforce.h"
// Note: this header for simplify HIP and CUDA type string
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
#include "paddle/fluid/platform/device/gpu/gpu_types.h"
#endif

COMMON_DECLARE_int32(call_stack_level);

namespace paddle {
namespace platform {
using namespace ::phi::enforce;  // NOLINT
using ::common::demangle;
using ::common::enforce::EnforceNotMet;

/** HELPER MACROS AND FUNCTIONS **/

#ifndef PADDLE_MAY_THROW
#define PADDLE_MAY_THROW noexcept(false)
#endif

/*
 * Summary: This macro is used to check whether op has specified
 * Input or Output Variables. Because op's Input and Output
 * checking are written similarly, so abstract this macro.
 *
 * Parameters:
 *     __EXPR: (bool), the bool expression
 *     __ROLE: (string), Input or Output
 *     __NAME: (string), Input or Output name
 *     __OP_TYPE: (string), the op type
 *
 * Examples:
 *    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Mul");
 */
#define OP_INOUT_CHECK(__EXPR, __ROLE, __NAME, __OP_TYPE)                    \
  do {                                                                       \
    PADDLE_ENFORCE_EQ(                                                       \
        __EXPR,                                                              \
        true,                                                                \
        phi::errors::NotFound(                                               \
            "No %s(%s) found for %s operator.", __ROLE, __NAME, __OP_TYPE)); \
  } while (0)

/** OTHER EXCEPTION AND ENFORCE **/

struct EOFException : public std::exception {
  std::string err_str_;
  EOFException(const char* err_msg, const char* file, int line) {
    err_str_ = paddle::string::Sprintf("%s at [%s:%d]", err_msg, file, line);
  }

  const char* what() const noexcept override { return err_str_.c_str(); }
};

#define PADDLE_THROW_EOF()                             \
  do {                                                 \
    HANDLE_THE_ERROR                                   \
    throw paddle::platform::EOFException(              \
        "There is no next data.", __FILE__, __LINE__); \
    END_HANDLE_THE_ERROR                               \
  } while (0)

#define PADDLE_THROW_BAD_ALLOC(...)                                      \
  do {                                                                   \
    HANDLE_THE_ERROR                                                     \
    throw ::paddle::memory::allocation::BadAlloc(                        \
        phi::ErrorSummary(__VA_ARGS__).to_string(), __FILE__, __LINE__); \
    END_HANDLE_THE_ERROR                                                 \
  } while (0)

}  // namespace platform
}  // namespace paddle
