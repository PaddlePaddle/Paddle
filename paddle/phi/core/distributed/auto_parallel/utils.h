/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

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

#include <algorithm>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/phi/core/enforce.h"

namespace phi {
namespace distributed {
namespace auto_parallel {

// struct Indent {
//   Indent(int &level) : level(level) { ++level; }
//   ~Indent() { --level; }
//   int &level;
// };

// inline std::string str_indent(std::string& str, cur_indent) {
//   string spaces(cur_indent, " ");
//   return str + std::string(cur_indent, " ");
// }

template <class T>
bool has_duplicates(const std::vector<T>& vec) {
  std::unordered_map<T, int> map;
  for (const auto& i : vec) {
    ++map[i];
    if (map[i] > 1) return true;
  }
  return false;
}

inline int64_t canonical_dim(int dim, int ndim) {
  PADDLE_ENFORCE_EQ(
      dim >= -ndim && dim < ndim,
      true,
      errors::InvalidArgument(
          "Dimension %d is outside of [-%d, %d).", dim, ndim, ndim));
  if (dim < 0) {
    return dim + ndim;
  }
  return dim;
}

// Refer to https://stackoverflow.com/a/5289170
template <typename Range, typename Value = typename Range::value_type>
std::string str_join(Range const& elements,
                     const std::string& delimiter = ",") {
  std::ostringstream os;
  auto b = std::begin(elements), e = std::end(elements);

  if (b != e) {
    std::copy(b, prev(e), std::ostream_iterator<Value>(os, delimiter.c_str()));
    b = prev(e);
  }
  if (b != e) {
    os << *b;
  }

  return os.str();
}

inline std::string str_join(std::map<std::string, bool> const& elements,
                            const std::string& delimiter = ",") {
  std::string str;
  for (const auto& item : elements) {
    str += item.first + ": " + std::to_string(item.second) + delimiter;
  }
  return str.substr(0, str.size() - 1);
}

// Refer to https://stackoverflow.com/a/46931770
inline std::vector<std::string> str_split(std::string const& input,
                                          const std::string& delimiter = ",") {
  size_t pos_start = 0, pos_end, delim_len = delimiter.length();
  std::string token;
  std::vector<std::string> output;
  while ((pos_end = input.find(delimiter, pos_start)) != std::string::npos) {
    token = input.substr(pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    output.push_back(token);
  }
  output.push_back(input.substr(pos_start));
  return output;
}

// Refer to https://stackoverflow.com/a/29200671/2358969
template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 2) {
  std::ostringstream out;
  out.precision(n);
  out << std::fixed << a_value;
  return out.str();
}

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace phi
