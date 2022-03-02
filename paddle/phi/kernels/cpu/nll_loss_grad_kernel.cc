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

#include "paddle/phi/kernels/nll_loss_grad_kernel.h"

#include <memory>
#include <string>
#include "paddle/fluid/operators/math.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {
template <typename T>
static void nll_loss_grad_1D(T* dx_data,
                             const T* dout_data,
                             const int64_t* label_data,
                             const T* weight_data,
                             const T* total_weight_data,
                             const int64_t batch_size,
                             const int64_t n_classes,
                             const std::string reduction,
                             const int64_t ignore_index) {
  if (reduction == "none") {
    for (int i = 0; i < batch_size; i++) {
      const auto cur_label = label_data[i];
      if (cur_label == ignore_index) {
        continue;
      }
      const auto cur_weight =
          weight_data ? weight_data[cur_label] : static_cast<T>(1);
      dx_data[i * n_classes + cur_label] = -dout_data[i] * cur_weight;
    }
    return;
  }

  const T dout_val = *dout_data;
  const T total_weight_val = *total_weight_data;
  for (int i = 0; i < batch_size; i++) {
    const auto cur_label = label_data[i];
    if (cur_label == ignore_index) {
      continue;
    }
    const auto cur_weight =
        weight_data ? weight_data[cur_label] : static_cast<T>(1);
    dx_data[i * n_classes + cur_label] = -dout_val * cur_weight;
    if (reduction == "mean") {
      dx_data[i * n_classes + cur_label] /= total_weight_val;
    }
  }
}

template <typename T>
static void nll_loss_grad_2D(T* dx_data,
                             const T* dout_data,
                             const int64_t* label_data,
                             const T* weight_data,
                             const T* total_weight_data,
                             const int64_t batch_size,
                             const int64_t n_classes,
                             const int64_t in_dim2,
                             const int64_t in_dim3,
                             const std::string& reduction,
                             const int64_t ignore_index) {
  const auto map_size = in_dim2 * in_dim3;
  const auto sample_size = n_classes * map_size;

  if (reduction == "none") {
    for (int i = 0; i < batch_size; i++) {
      for (int h = 0; h < in_dim2; h++) {
        for (int w = 0; w < in_dim3; w++) {
          const auto index = i * map_size + h * in_dim3 + w;
          const auto cur_label = label_data[index];
          if (cur_label == ignore_index) {
            continue;
          }
          const auto cur_weight =
              weight_data ? weight_data[cur_label] : static_cast<T>(1);
          dx_data[i * sample_size + cur_label * map_size + h * in_dim3 + w] =
              -cur_weight * dout_data[index];
        }
      }
    }
    return;
  }

  const T dout_val = *dout_data;
  const T total_weight_val = *total_weight_data;
  for (int i = 0; i < batch_size; i++) {
    for (int h = 0; h < in_dim2; h++) {
      for (int w = 0; w < in_dim3; w++) {
        const auto index = i * map_size + h * in_dim3 + w;
        const auto cur_label = label_data[index];
        if (cur_label == ignore_index) {
          continue;
        }
        const auto cur_weight =
            weight_data ? weight_data[cur_label] : static_cast<T>(1);
        const auto dx_index =
            i * sample_size + cur_label * map_size + h * in_dim3 + w;
        dx_data[dx_index] = -dout_val * cur_weight;
        if (reduction == "mean") {
          dx_data[dx_index] /= total_weight_val;
        }
      }
    }
  }
}

template <typename T, typename Context>
void NllLossGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& labels,
                       const DenseTensor& total_weight,
                       paddle::optional<const DenseTensor&> weight,
                       const DenseTensor& d_out,
                       int64_t ignore_index,
                       const std::string& reduction,
                       DenseTensor* dx) {
  auto dx_data = dev_ctx.template Alloc<T>(dx);
  auto dout_data = d_out.data<T>();
  auto label_data = labels.data<int64_t>();
  auto weight_data = weight.get_ptr() ? weight.get_ptr()->data<T>() : nullptr;
  auto total_weight_data = total_weight.data<T>();
  memset(dx_data, 0, dx->numel() * sizeof(T));

  const auto x_dims = x.dims();
  const auto batch_size = x_dims[0];
  const auto n_classes = x_dims[1];

  if (x_dims.size() == 2) {
    nll_loss_grad_1D(dx_data,
                     dout_data,
                     label_data,
                     weight_data,
                     total_weight_data,
                     batch_size,
                     n_classes,
                     reduction,
                     ignore_index);
  } else if (x_dims.size() == 4) {
    const auto in_dim2 = x_dims[2];
    const auto in_dim3 = x_dims[3];
    nll_loss_grad_2D(dx_data,
                     dout_data,
                     label_data,
                     weight_data,
                     total_weight_data,
                     batch_size,
                     n_classes,
                     in_dim2,
                     in_dim3,
                     reduction,
                     ignore_index);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    nll_loss_grad, CPU, ALL_LAYOUT, phi::NllLossGradKernel, float, double) {}
