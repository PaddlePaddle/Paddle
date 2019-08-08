// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <stdarg.h>  // For va_start, etc.
#include <algorithm>
#include <cstring>
#include <memory>  // For std::unique_ptr
#include <sstream>
#include <string>
#include <vector>

namespace paddle {
namespace lite {

static std::string string_format(const std::string fmt_str, ...) {
  /* Reserve two times as much as the length of the fmt_str */
  int final_n, n = (static_cast<int>(fmt_str.size())) * 2;
  std::unique_ptr<char[]> formatted;
  va_list ap;
  while (1) {
    formatted.reset(
        new char[n]); /* Wrap the plain char array into the unique_ptr */
    std::strcpy(&formatted[0], fmt_str.c_str());  // NOLINT
    va_start(ap, fmt_str);
    final_n = vsnprintf(&formatted[0], n, fmt_str.c_str(), ap);
    va_end(ap);
    if (final_n < 0 || final_n >= n)
      n += abs(final_n - n + 1);
    else
      break;
  }
  return std::string(formatted.get());
}

template <typename T>
static std::string to_string_with_precision(const T& v, const int n = 6) {
  std::stringstream ss;
  ss.precision(n);
  ss << std::fixed << v;
  return ss.str();
}

template <typename T>
std::string Join(const std::vector<T>& vec, const std::string& delim) {
  if (vec.empty()) return "";

  std::stringstream ss;
  for (size_t i = 0; i < vec.size() - 1; i++) ss << vec[i] << delim;
  if (!vec.empty()) {
    ss << vec.back();
  }

  return ss.str();
}

static std::string Repr(const std::string& x) { return "\"" + x + "\""; }

static std::string Repr(const std::vector<std::string>& v) {
  std::vector<std::string> tmp;
  std::transform(v.begin(), v.end(), std::back_inserter(tmp),
                 [](const std::string& x) { return Repr(x); });
  return "{" + Join(tmp, ",") + "}";
}

static std::vector<std::string> Split(const std::string& original,
                                      const std::string& separator) {
  std::vector<std::string> results;
  std::string::size_type pos1, pos2;
  pos2 = original.find(separator);
  pos1 = 0;
  while (std::string::npos != pos2) {
    results.push_back(original.substr(pos1, pos2 - pos1));
    pos1 = pos2 + separator.size();
    pos2 = original.find(separator, pos1);
  }
  if (pos1 != original.length()) {
    results.push_back(original.substr(pos1));
  }
  return results;
}

}  // namespace lite
}  // namespace paddle
