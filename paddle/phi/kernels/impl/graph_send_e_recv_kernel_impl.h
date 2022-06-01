// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

struct BroadCastInfo {
  bool use_bcast;
  // l_offset[i] indicates the start position of tensor lhs that required to
  // compute the i-th element in output, so as r_offset[i].
  std::vector<int64_t> l_offset, r_offset;
  int64_t l_len, r_len, out_len, reduce_size;
};

BroadCastInfo CalcBCastInfo(const phi::DDim& l_dims, const phi::DDim& r_dims);

}  // namespace phi
