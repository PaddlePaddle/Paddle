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

#include "paddle/pten/core/dense_tensor.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/device_context.h"
namespace pten {

using CPUContext = paddle::platform::CPUDeviceContext;
namespace detail {

template <typename T>
void SplitImpl(const CPUContext& ctx,
               const pten::DenseTensor& input,
               const std::vector<const pten::DenseTensor*>& ref_inputs,
               const int axis,
               std::vector<DenseTensor*>* outputs) {
  // NOTE(zhiqiu): split a tensor of shape [0,3,4] at axis=1, result in 3
  // tensors of shape [0,1,4]
  if (input.numel() == 0) {
    return;
  }

  // TODO(zcd): Add input data validity checking
  size_t num = outputs->size();

  int input_rows = 1;
  auto dim_0 = ref_inputs[0]->dims();
  for (int i = 0; i < axis; ++i) {
    input_rows *= dim_0[i];
  }

  int input_cols = 0;

  std::vector<int64_t> output_cols(outputs->size());
  for (size_t i = 0; i < num; ++i) {
    int t_cols = ref_inputs[i]->numel() / input_rows;
    input_cols += t_cols;
    output_cols[i] = t_cols;
  }
  auto cpu_place = BOOST_GET_CONST(paddle::platform::CPUPlace, ctx.GetPlace());

  // computation
  for (int k = 0; k < input_rows; ++k) {
    const T* src_ptr = input.data<T>() + k * input_cols;
    int col_idx = 0;
    for (size_t j = 0; j < num; ++j) {
      int col_len = output_cols[j];
      auto* out_tensor = outputs->at(j);
      if (out_tensor != nullptr) {
        T* dst_ptr = out_tensor->mutable_data<T>() + k * col_len;
        paddle::memory::Copy(cpu_place,
                             dst_ptr,
                             cpu_place,
                             src_ptr + col_idx,
                             sizeof(T) * col_len);
      }
      col_idx += col_len;
    }
  }
}

}  // namespace detail

}  // namespace pten
