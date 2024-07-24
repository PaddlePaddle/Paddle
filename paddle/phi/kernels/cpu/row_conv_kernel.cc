// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <vector>
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/mixed_vector.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"

namespace phi {

template <typename T, typename Context>
void RowConvKernel(const Context &dev_ctx,
                   const DenseTensor &x_in,
                   const DenseTensor &filter_in,
                   DenseTensor *out) {
  auto *x = &x_in;
  auto *filter = &filter_in;

  dev_ctx.template Alloc<T>(out);

  bool is_tensor = x->lod().empty();
  int batch_size = 0;
  if (is_tensor) {
    batch_size = static_cast<int>(x->dims()[0]);
  } else {
    batch_size = static_cast<int>(x->lod()[0].size() - 1);
  }
  phi::Vector<size_t> batch_indices(batch_size + 1);
  int input_dim = 0;
  int timesteps = 0;
  if (is_tensor) {
    for (int i = 0; i < batch_size + 1; i++) {
      batch_indices[i] = i;
    }
    input_dim = static_cast<int>(x->dims()[2]);
    timesteps = static_cast<int>(x->dims()[1]);
  } else {
    batch_indices = x->lod()[0];
    input_dim = static_cast<int>(x->dims()[1]);
  }
  size_t num_sequence = batch_indices.size() - 1;

  auto future_context = filter->dims()[0];
  auto weights = EigenMatrix<T>::From(*filter);

  for (size_t i = 0; i < num_sequence; i++) {
    int start = static_cast<int>(batch_indices[i]);
    int end = static_cast<int>(batch_indices[i + 1]);
    int current_timesteps = 0;
    if (is_tensor) {
      current_timesteps = timesteps;
    } else {
      current_timesteps = end - start;
    }
    // int current_timesteps = end - start;
    phi::DenseTensor cur_input_sequence =
        x->Slice(start, end);  // Current input sequence
    cur_input_sequence =
        cur_input_sequence.Resize({current_timesteps, input_dim});

    phi::DenseTensor cur_output_sequence =
        out->Slice(start, end);  // Current output sequence
    cur_output_sequence =
        cur_output_sequence.Resize({current_timesteps, input_dim});

    auto cip_seq = EigenMatrix<T>::From(cur_input_sequence);
    auto cot_seq = EigenMatrix<T>::From(cur_output_sequence);

    for (int k = 0; k < current_timesteps;
         k++) {  // For different time steps in the same sequence
      for (int w = 0; (w < future_context) && ((k + w) < current_timesteps);
           w++) {
        for (int d = 0; d < input_dim; d++) {
          if (w == 0) {
            cot_seq(k, d) = weights(w, d) * cip_seq(k + w, d);
          } else {
            cot_seq(k, d) += weights(w, d) * cip_seq(k + w, d);
          }
        }
      }
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(row_conv, CPU, ALL_LAYOUT, phi::RowConvKernel, float) {}
