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

namespace phi {

struct BroadCastInfo {
  bool use_bcast;
  // x_offset[i] indicates the start position of tensor x that required to
  // compute the i-th element in output, so as e_offset[i].
  std::vector<int64_t> x_offset, e_offset;
  int64_t x_len, e_len, out_len, reduce_size;
};

bool UseBroadCast(const phi::DDim& x_dims, const phi::DDim& e_dims) {
  if (x_dims.size() != e_dims.size()) {
    return True;
  }
  for (int i = 0; i < x_dims.size(); i++) {
    if (x_dims[i] != e_dims[i]) {
      return True;
    }
  }
  return False;
}

BroadCastInfo CaclBCastInfo(const phi::DDim& x_dims, const phi::DDim& e_dims) {
  BroadCastInfo binfo;
  binfo.use_bcast = UseBroadCast(x_dims, e_dims);
  binfo.x_len = 1;
  binfo.e_len = 1;
  for (int i = 1; i < x_dims.size(); i++) {
    binfo.x_len *= x_dims[i];
  }
  for (int i = 1; i < e_dims.size(); i++) {
    binfo.e_len *= e_dims[i];
  }
  // TODO(daisiming): Whether to add dot.
  binfo.reduce_size = 1;
  if (binfo.use_bcast) {
    const int max_dim = std::max(x_dims.size(), e_dims.size()) - 1;
    int stride_x = 1, stride_e = 1;
    binfo.x_offset.emplace_back(0);
    binfo.e_offset.emplace_back(0);
    int out_len = 1;
    for (int i = 0; i < max_dim; i++) {
      // Iterate the axis from back to front.
      const int dl =
          (x_dims.size() - 1 - i < 1) ? 1 : x_dims[x_dims.size() - 1 - i];
      const int dr =
          (e_dims.size() - 1 - i < 1) ? 1 : e_dims[e_dims.size() - 1 - i];
      for (int j = 0; j < std::max(dl, dr); j++) {
        for (int k = 0; k < out_len; k++) {
          binfo.x_offset.emplace_back(binfo.x_offset[k] +
                                      j * (j < dl) * stride_x);
          binfo.e_offset.emplace_back(binfo.e_offset[k] +
                                      j * (j < dr) * stride_e);
        }
      }
      out_len *= std::max(dl, dr);
      stride_x *= dl;
      stride_e *= dr;
    }
    binfo.out_len = out_len;
  } else {
    binfo.out_len = binfo.x_len;
  }
  return binfo;
}

}  // namespace phi
