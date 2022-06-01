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

inline bool UseBroadCast(const phi::DDim& l_dims, const phi::DDim& r_dims) {
  if (l_dims.size() != r_dims.size()) {
    return true;
  }
  for (int i = 1; i < l_dims.size(); i++) {
    if (l_dims[i] != r_dims[i]) {
      return true;
    }
  }
  return false;
}

inline BroadCastInfo CalcBCastInfo(const phi::DDim& l_dims,
                                   const phi::DDim& r_dims) {
  BroadCastInfo binfo;
  binfo.use_bcast = UseBroadCast(l_dims, r_dims);
  binfo.l_len = 1;
  binfo.r_len = 1;
  for (int i = 1; i < l_dims.size(); i++) {
    binfo.l_len *= l_dims[i];
  }
  for (int i = 1; i < r_dims.size(); i++) {
    binfo.r_len *= r_dims[i];
  }
  // TODO(daisiming): Whether to add dot.
  binfo.reduce_size = 1;
  if (binfo.use_bcast) {
    const int max_dim = std::max(l_dims.size(), r_dims.size()) - 1;
    int stride_l = 1, stride_r = 1;
    binfo.l_offset.emplace_back(0);
    binfo.r_offset.emplace_back(0);
    int out_len = 1;
    for (int i = 0; i < max_dim; i++) {
      // Iterate the axis from back to front.
      const int dl =
          (l_dims.size() - 1 - i < 1) ? 1 : l_dims[l_dims.size() - 1 - i];
      const int dr =
          (r_dims.size() - 1 - i < 1) ? 1 : r_dims[r_dims.size() - 1 - i];
      for (int j = 0; j < std::max(dl, dr); j++) {
        for (int k = 0; k < out_len; k++) {
          binfo.l_offset.emplace_back(binfo.l_offset[k] +
                                      j * (j < dl) * stride_l);
          binfo.r_offset.emplace_back(binfo.r_offset[k] +
                                      j * (j < dr) * stride_r);
        }
      }
      out_len *= std::max(dl, dr);
      stride_l *= dl;
      stride_r *= dr;
    }
    binfo.out_len = out_len;
  } else {
    binfo.out_len = binfo.l_len;
  }
  return binfo;
}

}  // namespace phi
