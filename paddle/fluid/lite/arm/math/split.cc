// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/lite/arm/math/split.h"
#include <algorithm>
#include "paddle/fluid/lite/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <>
void split_cpy<float>(const float* din, float* dout, int num) {
  int cnt = num >> 4;
  int remain = num % 16;
#pragma omp parallel for
  for (int i = 0; i < cnt; i++) {
    const float* din_ptr = din + (i << 4);
    float* dout_ptr = dout + (i << 4);

    float32x4_t din0 = vld1q_f32(din_ptr);
    float32x4_t din1 = vld1q_f32(din_ptr + 4);
    float32x4_t din2 = vld1q_f32(din_ptr + 8);
    float32x4_t din3 = vld1q_f32(din_ptr + 12);

    vst1q_f32(dout_ptr, din0);
    vst1q_f32(dout_ptr + 4, din1);
    vst1q_f32(dout_ptr + 8, din2);
    vst1q_f32(dout_ptr + 12, din3);
  }
  if (remain > 0) {
    const float* din_ptr = din + (cnt << 4);
    float* dout_ptr = dout + (cnt << 4);
    for (int i = 0; i < remain; i++) {
      *dout_ptr = *din_ptr;
      dout_ptr++;
      din_ptr++;
    }
  }
}

template <>
void split<float>(const float* din, std::vector<lite::Tensor*>* dout,
                  const int axis, const std::vector<int>& in_strides) {
  int input_offset = 0;
  for (auto out : *dout) {
    auto out_dim = out->dims();
    std::vector<int> out_strides(out_dim.size());
    out_strides[out_dim.size() - 1] = out_dim[out_dim.size() - 1];
    for (int i = out_dim.size() - 2; i >= 0; --i) {
      out_strides[i] = out_strides[i + 1] * out_dim[i];
    }

    float* out_data = out->mutable_data<float>();
    int before = out_strides[0] / out_strides[axis];
    int in_after = in_strides[axis];
    int out_after = out_strides[axis];

    for (int i = 0; i < before; ++i) {
      split_cpy(din + input_offset + i * in_after, out_data + i * out_after,
                out_after);
    }
    input_offset += out_strides[axis];
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
