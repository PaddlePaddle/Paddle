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
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace distributed {
namespace auto_parallel {

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
  PADDLE_ENFORCE(dim >= -ndim);
  PADDLE_ENFORCE(dim < ndim);
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

}  // namespace auto_parallel
}  // namespace distributed
}  // namespace paddle
