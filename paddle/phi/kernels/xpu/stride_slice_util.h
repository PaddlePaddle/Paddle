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

#include <vector>

namespace phi {

inline bool is_strided_slice_special_case(const std::vector<int>& xshape,
                                          const std::vector<int>& starts,
                                          const std::vector<int>& ends,
                                          const std::vector<int>& strides) {
  // starts match {0, 0, ..., 0, 0} or {0, 0, ..., 0, 1}
  for (size_t i = 0; i < starts.size() - 1; i++) {
    if (starts[i] != 0) {
      return false;
    }
  }
  if (starts.back() != 0 && starts.back() != 1) {
    return false;
  }
  // xshape match ends
  if (xshape != ends) {
    return false;
  }
  // strides match {1, 1, ..., 1, 2}
  for (size_t i = 0; i < strides.size() - 1; i++) {
    if (strides[i] != 1) {
      return false;
    }
  }
  if (strides.back() != 2) {
    return false;
  }
  // last dim of xshape is even number
  if (xshape.back() % 2 != 0) {
    return false;
  }
  return true;
}

}  // namespace phi
