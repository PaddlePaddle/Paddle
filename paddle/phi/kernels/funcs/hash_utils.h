// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/enforce.h"
namespace phi {
namespace funcs {

inline void HashOutputSize(const phi::DDim& in_dims,
                           std::vector<int64_t>& out_dims,  // NOLINT
                           int num_hash) {
  out_dims.reserve(in_dims.size() + 1);
  // copy all dims except the last one
  for (int i = 0u; i != in_dims.size() - 1; ++i) {
    out_dims.emplace_back(in_dims[i]);
  }
  out_dims.emplace_back(num_hash);
  // keep the last dim to 1
  out_dims.emplace_back(1);
}
}  // namespace funcs
}  // namespace phi
