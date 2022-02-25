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

namespace phi {
namespace kps {

#ifdef PADDLE_WITH_XPU_KP
struct dim3 {
  int x;
  int y;
  int z;

  explicit inline dim3(int split_x, int split_y = 1, int split_z = 1) {
    x = split_x;
    y = split_y;
    z = split_z;
  }
};
#endif

struct DimConfig {
  int split_num_x;
  int split_num_y;
  int split_num_z;
  int deal_size_x;
  int deal_size_y;
  int deal_size_z;
  int rem_x;
  int rem_y;
  int rem_z;

  HOSTDEVICE explicit inline DimConfig(int split_x,
                                       int split_y,
                                       int split_z,
                                       int size_x,
                                       int size_y,
                                       int size_z) {
    split_num_x = split_x;
    split_num_y = split_y;
    split_num_z = split_z;
    deal_size_x = size_x;
    deal_size_y = size_y;
    deal_size_z = size_z;
  }

  HOSTDEVICE void SetRem(int rem_nx, int rem_ny, int rem_nz) {
    rem_x = rem_nx;
    rem_y = rem_ny;
    rem_z = rem_nz;
  }
};

}  // namespace kps
}  // namespace phi
