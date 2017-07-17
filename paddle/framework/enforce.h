/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
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
#include <paddle/string/printf.h>
#include <exception>
#include <sstream>

namespace paddle {
namespace framework {

/**
 * @brief Enforce exception. Inherits std::exception
 *
 * All enforce condition not met, will throw an EnforceNotMet exception.
 */
class EnforceNotMet : public std::exception {
 public:
  EnforceNotMet(const std::string& msg, const char* file, int fileline) {
    std::ostringstream sout;
    sout << msg << " at [" << file << ":" << fileline << "];";
    all_msg_ = sout.str();
  }

  const char* what() const noexcept override { return all_msg_.c_str(); }

 private:
  std::string all_msg_;
};

// From https://stackoverflow.com/questions/30130930/
// __buildin_expect is in C++ 11 standard. Since the condition which enforced
// should be true in most situation, it will make the compiler generate faster
// code by adding `UNLIKELY` macro.
#define UNLIKELY(condition) __builtin_expect(static_cast<bool>(condition), 0)

/**
 * @brief Throw a EnforceNotMet exception, automatically filled __FILE__ &
 * __LINE__
 *
 * This macro take __VA_ARGS__, user can pass any type if that type can
 * serialize to std::ostream
 */
#define PADDLE_THROW(...)                                            \
  do {                                                               \
    throw ::paddle::framework::EnforceNotMet(                        \
        ::paddle::string::Sprintf(__VA_ARGS__), __FILE__, __LINE__); \
  } while (0)

/**
 * @brief Enforce a condition, otherwise throw an EnforceNotMet
 */
#define PADDLE_ENFORCE(condition, ...) \
  do {                                 \
    if (UNLIKELY(!(condition))) {      \
      PADDLE_THROW(__VA_ARGS__);       \
    }                                  \
  } while (0)

}  // namespace framework
}  // namespace paddle
