// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <exception>
#include <string>

#include "paddle/utils/string/printf.h"

#if !defined(_WIN32)
#define UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)
#else
// there is no equivalent intrinsics in msvc.
#define UNLIKELY(condition) (condition)
#endif
template <typename T>
inline bool is_error(const T& stat) {
  return !stat;
}

namespace pir {
class IrNotMetException : public std::exception {
 public:
  explicit IrNotMetException(const std::string& str) : err_str_(str) {}

  const char* what() const noexcept override { return err_str_.c_str(); }

 private:
  std::string err_str_;
};

#define IR_THROW(...)                                                     \
  do {                                                                    \
    try {                                                                 \
      throw pir::IrNotMetException(                                       \
          paddle::string::Sprintf("Error occured at: %s:%d :\n%s",        \
                                  __FILE__,                               \
                                  __LINE__,                               \
                                  paddle::string::Sprintf(__VA_ARGS__))); \
    } catch (const std::exception& e) {                                   \
      std::cout << e.what() << std::endl;                                 \
      throw;                                                              \
    }                                                                     \
  } while (0)

#define IR_ENFORCE(COND, ...)                                               \
  do {                                                                      \
    bool __cond__(COND);                                                    \
    if (UNLIKELY(is_error(__cond__))) {                                     \
      try {                                                                 \
        throw pir::IrNotMetException(                                       \
            paddle::string::Sprintf("Error occured at: %s:%d :\n%s",        \
                                    __FILE__,                               \
                                    __LINE__,                               \
                                    paddle::string::Sprintf(__VA_ARGS__))); \
      } catch (const std::exception& e) {                                   \
        std::cout << e.what() << std::endl;                                 \
        throw;                                                              \
      }                                                                     \
    }                                                                       \
  } while (0)

}  // namespace pir
