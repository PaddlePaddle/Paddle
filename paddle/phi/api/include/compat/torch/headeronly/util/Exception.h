// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <c10/macros/Macros.h>

#include <sstream>
#include <stdexcept>
#include <string>

#ifndef C10_UNLIKELY
#if defined(__GNUC__) || defined(__clang__)
#define C10_UNLIKELY(expr) (__builtin_expect(static_cast<bool>(expr), 0))
#else
#define C10_UNLIKELY(expr) (expr)
#endif
#endif

namespace c10 {

// Keep constexpr-friendly control flow when the check condition is constant
// under nvcc/hipcc, matching upstream headeronly behavior.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define C10_UNLIKELY_OR_CONST(e) e
#else
#define C10_UNLIKELY_OR_CONST(e) C10_UNLIKELY(e)
#endif

}  // namespace c10

#ifdef STRIP_ERROR_MESSAGES
#define STD_TORCH_CHECK_MSG(cond, type, ...) \
  (#cond #type " CHECK FAILED at " C10_STRINGIZE(__FILE__))
#else
namespace torch::headeronly::detail {

template <typename... Args>
inline std::string stdTorchCheckMsgImpl(const char* /*msg*/,
                                        const Args&... args) {
  std::ostringstream oss;
  ((oss << args), ...);
  return oss.str();
}

inline const char* stdTorchCheckMsgImpl(const char* msg) { return msg; }

inline const char* stdTorchCheckMsgImpl(const char* /*msg*/, const char* args) {
  return args;
}

}  // namespace torch::headeronly::detail

#define STD_TORCH_CHECK_MSG(cond, type, ...)               \
  (torch::headeronly::detail::stdTorchCheckMsgImpl(        \
      "Expected " #cond                                    \
      " to be true, but got false.  "                      \
      "(Could this error message be improved?  If so, "    \
      "please report an enhancement request to PyTorch.)", \
      ##__VA_ARGS__))
#endif

#define STD_TORCH_CHECK(cond, ...)                                \
  if (C10_UNLIKELY_OR_CONST(!(cond))) {                           \
    throw std::runtime_error(STD_TORCH_CHECK_MSG(cond,            \
                                                 "",              \
                                                 __func__,        \
                                                 ", ",            \
                                                 __FILE__,        \
                                                 ":",             \
                                                 __LINE__,        \
                                                 ", ",            \
                                                 ##__VA_ARGS__)); \
  }
