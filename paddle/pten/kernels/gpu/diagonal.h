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

#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
#include "paddle/pten/backends/gpu/gpu_context.h"

namespace pten {
template <typename T, int X_DIM_SIZE, int OUT_DIM_SIZE>
__global__ void Diagonal(const T* data1,
                         T* data2,
                         const int64_t offset_,
                         int64_t axis1_,
                         int64_t axis2_,
                         int64_t* x_stride,
                         int64_t* out_stride,
                         int64_t numel,
                         bool is_grad) {
  CUDA_KERNEL_LOOP(idx, numel) {
    int64_t idx_dim[X_DIM_SIZE] = {0};
    int64_t temp = 0;
    for (size_t i = 0; i < X_DIM_SIZE - 1; i++) {
      idx_dim[i] = (idx - temp) / x_stride[i];
      temp = temp + idx_dim[i] * x_stride[i];
    }
    idx_dim[X_DIM_SIZE - 1] = idx - temp;

    int64_t axis1_dim = idx_dim[axis1_];
    int64_t axis2_dim = idx_dim[axis2_];

    int64_t out_dim[OUT_DIM_SIZE] = {0};
    int temp_pos = 0;
    for (int i = 0; i < X_DIM_SIZE; i++) {
      if (i != axis1_ && i != axis2_) {
        out_dim[temp_pos] = idx_dim[i];
        temp_pos++;
      }
    }
    bool flag = false;
    if (offset_ == 0 && axis1_dim == axis2_dim) {
      out_dim[temp_pos] = axis1_dim;
      flag = true;
    } else if (offset_ > 0 && (axis1_dim + offset_) == axis2_dim) {
      out_dim[temp_pos] = axis1_dim;
      flag = true;
    } else if (offset_ < 0 && (axis1_dim + offset_) == axis2_dim) {
      out_dim[temp_pos] = axis2_dim;
      flag = true;
    }
    if (!is_grad) {
      if (flag) {
        int64_t idx_output = 0;
        for (size_t i = 0; i < OUT_DIM_SIZE - 1; i++) {
          idx_output = idx_output + out_dim[i] * out_stride[i];
        }
        idx_output = idx_output + out_dim[OUT_DIM_SIZE - 1];
        data2[idx_output] = data1[idx];
      }
    } else {
      if (flag) {
        int64_t idx_output = 0;
        for (size_t i = 0; i < OUT_DIM_SIZE - 1; i++) {
          idx_output = idx_output + out_dim[i] * out_stride[i];
        }
        idx_output = idx_output + out_dim[OUT_DIM_SIZE - 1];
        data2[idx] = data1[idx_output];
      } else {
        data2[idx] = static_cast<T>(0);
      }
    }
  }
}
}  // namespace pten
