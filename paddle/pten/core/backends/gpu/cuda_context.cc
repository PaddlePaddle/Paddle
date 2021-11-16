/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License. */

#include "paddle/pten/core/backends/gpu/cuda_context.h"

namespace pten {

dim3 CUDAContext::GetCUDAMaxGridDimSize() const {
  dim3 ret;
  ret.x = max_grid_dim_x_;
  ret.y = max_grid_dim_y_;
  ret.z = max_grid_dim_z_;
  return ret;
}
}
