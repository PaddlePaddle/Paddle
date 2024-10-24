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
void RowConvGradKernel(const Context &dev_ctx,
                       const DenseTensor &x_in,
                       const DenseTensor &filter_in,
                       const DenseTensor &out_grad,
                       DenseTensor *x_grad,
                       DenseTensor *filter_grad) {
  auto *x = &x_in;
  auto *filter = &filter_in;
  auto *d_out = &out_grad;
  auto *dx = x_grad;
  auto *d_filter = filter_grad;

  auto &x_lod = x->lod();
  bool is_tensor = x_lod.empty();
  int batch_size = 0;
  if (is_tensor) {
    batch_size = static_cast<int>(x->dims()[0]);
  } else {
    batch_size = static_cast<int>(x->lod()[0].size() - 1);
  }
  phi::Vector<size_t> batch_indices(batch_size + 1);
  int timesteps = 0;
  int input_dim = 0;
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
  if (d_filter) {
    dev_ctx.template Alloc<T>(d_filter);
    auto dweights =
        EigenMatrix<T>::From(*d_filter);  // Gradient of weight matrix
    dweights.setZero();

    for (size_t i = 0; i < num_sequence; i++) {  // For different sequences
      int start = static_cast<int>(batch_indices[i]);
      int end = static_cast<int>(batch_indices[i + 1]);

      int current_timesteps = 0;
      if (is_tensor) {
        current_timesteps = timesteps;
      } else {
        current_timesteps = end - start;
      }
      phi::DenseTensor cur_input =
          x->Slice(start, end);  // Current input sequence
      cur_input = cur_input.Resize({current_timesteps, input_dim});
      phi::DenseTensor cur_doutput =
          d_out->Slice(start, end);  // Current output grad sequence
      cur_doutput = cur_doutput.Resize({current_timesteps, input_dim});
      auto cur_ip = EigenMatrix<T>::From(cur_input);
      auto cur_dout = EigenMatrix<T>::From(cur_doutput);
      for (int k = 0; k < current_timesteps;
           k++) {  // For different time steps in the same sequence
        for (int w = 0; (w < future_context) && ((k + w) < current_timesteps);
             w++) {
          // For dweights (Updating the gradient of weight matrix)
          for (int d = 0; d < input_dim; d++) {
            dweights(w, d) += cur_ip(k + w, d) * cur_dout(k, d);
          }
        }
      }
    }
  }

  if (dx) {
    dev_ctx.template Alloc<T>(dx);
    auto weights = EigenMatrix<T>::From(*filter);
    for (size_t i = 0; i < num_sequence; i++) {  // For different sequences
      int start = static_cast<int>(batch_indices[i]);
      int end = static_cast<int>(batch_indices[i + 1]);

      int current_timesteps = 0;
      if (is_tensor) {
        current_timesteps = timesteps;
      } else {
        current_timesteps = end - start;
      }

      phi::DenseTensor cur_doutput =
          d_out->Slice(start, end);  // Current output grad sequence
      cur_doutput = cur_doutput.Resize({current_timesteps, input_dim});
      phi::DenseTensor cur_dinput =
          dx->Slice(start, end);  // Current input grad sequence
      cur_dinput = cur_dinput.Resize({current_timesteps, input_dim});

      auto cur_dout = EigenMatrix<T>::From(cur_doutput);
      auto cur_dip = EigenMatrix<T>::From(cur_dinput);
      cur_dip.setZero();

      for (int k = 0; k < current_timesteps;
           k++) {  // For different time steps in the same sequence
        for (int w = 0; (w < future_context) && ((k + w) < current_timesteps);
             w++) {
          // For dinput (Updating the gradient wrt input)
          for (int d = 0; d < input_dim; d++) {
            cur_dip(k + w, d) += weights(w, d) * cur_dout(k, d);
          }
        }
      }
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    row_conv_grad, CPU, ALL_LAYOUT, phi::RowConvGradKernel, float) {}
