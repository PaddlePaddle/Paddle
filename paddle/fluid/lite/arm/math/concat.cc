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

#include "paddle/fluid/lite/arm/math/concat.h"
#include <algorithm>
#include <limits>
#include <memory>
#include "paddle/fluid/lite/arm/math/funcs.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

void concat_func(const std::vector<lite::Tensor *> &input, const int axis,
                 lite::Tensor *output) {
  size_t num = input.size();
  int rows = 1;
  auto dim_0 = input[0]->dims();
  for (int i = 0; i < axis; ++i) {
    rows *= dim_0[i];
  }
  int out_rows = rows, out_cols = 0;

  std::vector<int64_t> input_cols(input.size());
  for (int i = 0; i < num; ++i) {
    int t_cols = input[i]->numel() / rows;
    out_cols += t_cols;
    input_cols[i] = t_cols;
  }

  // computation
  for (int k = 0; k < out_rows; ++k) {
    float *dst_ptr = output->mutable_data<float>() + k * out_cols;
    int col_idx = 0;
    for (int j = 0; j < num; ++j) {
      int col_len = input_cols[j];
      const float *src_prt = input[j]->data<float>() + k * col_len;
      std::memcpy(dst_ptr + col_idx, src_prt, sizeof(float) * col_len);
      col_idx += col_len;
    }
  }
}

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
