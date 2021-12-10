// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#ifdef PADDLE_WITH_XPU2
#include "paddle/fluid/platform/eigen_ext.h"
#include "xpu/kernel/cluster_header.h"
#include "xpu/kernel/debug.h"
#include "xpu/kernel/math.h"
#endif

namespace paddle {
namespace operators {
namespace kernel_primitives {

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

  explicit inline DimConfig(int split_x, int split_y, int split_z, int size_x,
                            int size_y, int size_z, int rem_size_x = 0,
                            int rem_size_y = 0, int rem_size_z = 0) {
    split_num_x = split_x;
    split_num_y = split_y;
    split_num_z = split_z;
    deal_size_x = size_x;
    deal_size_y = size_y;
    deal_size_z = size_z;
    rem_x = rem_size_x;
    rem_y = rem_size_y;
    rem_z = rem_size_z;
  }

  __device__ explicit inline DimConfig(int split_x, int split_y, int split_z,
                                       int size_x, int size_y, int size_z,
                                       int rem_size_x = 0, int rem_size_y = 0,
                                       int rem_size_z = 0) {
    split_num_x = split_x;
    split_num_y = split_y;
    split_num_z = split_z;
    deal_size_x = size_x;
    deal_size_y = size_y;
    deal_size_z = size_z;
    rem_x = rem_size_x;
    rem_y = rem_size_y;
    rem_z = rem_size_z;
  }

  HOSTDEVICE void SetRem(int rem_nx, int rem_ny, int rem_nz) {
    rem_x = rem_nx;
    rem_y = rem_ny;
    rem_z = rem_nz;
  }
};

}  // namespace kernel_primitives
}  // namespace operators
}  // namespace paddle
