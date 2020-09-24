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
#include <sstream>
#include <string>
#include <type_traits>
#include <typeindex>
#include <vector>

namespace paddle {
namespace string {
inline std::ostream& operator<<(std::ostream& s, const std::type_index& t) {
  s << t.name();
  return s;
}

template <typename T,
          typename std::enable_if<!std::is_enum<T>::value, int>::type = 0>
inline std::string to_string(T v) {
  std::ostringstream sout;
  sout << v;
  return sout.str();
}

template <typename T,
          typename std::enable_if<std::is_enum<T>::value, int>::type = 0>
inline std::string to_string(T v) {
  return std::to_string(static_cast<int>(v));
}

template <>
inline std::string to_string(std::type_index t) {
  return t.name();
}

// Faster std::string/const char* type
template <>
inline std::string to_string(std::string v) {
  return v;
}

template <>
inline std::string to_string(const char* v) {
  return std::string(v);
}

}  // namespace string
}  // namespace paddle
