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

#include "paddle/pten/backends/cpu/cpu_context.h"
#include "paddle/pten/core/dense_tensor.h"

namespace pten {

/*
 * \brief Concatenate the input tensors along the dimension axis.
 *  TODO(zcd): maybe it needs to be more detailed.
 *  Examples:
 *     Input[0] = [[1,2],[3,4]]
 *     Input[1] = [[5,6]]
 *     axis = 0
 *
 *     Output = [[1,2],
 *               [3,4],
 *               [5,6]]
 */

template <typename T, typename Context>
void ConcatImpl(const Context& context,
                const std::vector<DenseTensor>& input,
                int axis,
                DenseTensor* output) {
  // TODO(zcd): Add input data validity checking
  size_t num = input.size();

  int64_t rows = 1;
  auto dim_0 = input[0].dims();
  for (int i = 0; i < axis; ++i) {
    rows *= dim_0[i];
  }
  int64_t out_rows = rows, out_cols = 0;

  std::vector<int64_t> input_cols(input.size());
  for (size_t i = 0; i < num; ++i) {
    int64_t t_cols = input[i].numel() / rows;
    out_cols += t_cols;
    input_cols[i] = t_cols;
  }
  auto cpu_place = context.GetPlace();

  // computation
  auto output_data = output->data<T>();
  int64_t col_idx = 0;
  for (size_t j = 0; j < num; ++j) {
    int64_t col_len = input_cols[j];
    auto input_data = input[j].data<T>();
    for (int64_t k = 0; k < out_rows; ++k) {
      paddle::memory::Copy(cpu_place,
                           output_data + k * out_cols + col_idx,
                           cpu_place,
                           input_data + k * col_len,
                           sizeof(T) * col_len);
    }
    col_idx += col_len;
  }
}

/*
 * \brief Split the input tensors along the dimension axis into outputs.
 *  TODO(zcd): maybe it needs to be more detailed.
 *  Examples:
 *     Input = [[1,2],
 *              [3,4],
 *              [5,6]]
 *     axis = 0
 *
 *     Output[0] = [[1,2],[3,4]]
 *     Output[1] = [[5,6]]
 */
template <typename T, typename Context>
void SplitImpl(const Context& context,
               const DenseTensor& input,
               const std::vector<const DenseTensor*>& ref_inputs,
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
  auto cpu_place = context.GetPlace();

  // computation
  for (int k = 0; k < input_rows; ++k) {
    const T* src_ptr = input.data<T>() + k * input_cols;
    int col_idx = 0;
    for (size_t j = 0; j < num; ++j) {
      int col_len = output_cols[j];
      auto* out_tensor = outputs->at(j);
      if (out_tensor != nullptr) {
        T* dst_ptr = out_tensor->data<T>() + k * col_len;
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

}  // namespace pten
