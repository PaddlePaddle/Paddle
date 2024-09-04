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

#include "paddle/phi/kernels/nll_loss_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
static void nll_loss_1D(T* out_data,
                        T* total_weight_data,
                        const T* x_data,
                        const int64_t* label_data,
                        const T* weight_data,
                        const int64_t batch_size,
                        const int64_t n_classes,
                        const std::string& reduction,
                        const int64_t ignore_index) {
  if (reduction == "none") {
    for (int64_t i = 0; i < batch_size; ++i) {
      const auto cur_label = label_data[i];
      if (cur_label == ignore_index) {
        out_data[i] = 0;
        continue;
      }
      PADDLE_ENFORCE_EQ(cur_label >= 0 && cur_label < n_classes,
                        true,
                        common::errors::InvalidArgument(
                            "Label value is out of range. "
                            "Expected label value in range of [0, %d), but "
                            "received value is %d.",
                            n_classes,
                            cur_label));

      const auto cur_weight =
          weight_data ? weight_data[cur_label] : static_cast<T>(1);
      out_data[i] = -x_data[i * n_classes + cur_label] * cur_weight;
    }
    return;
  }

  T output_val = 0;
  T total_weight_val = 0;

  for (int64_t i = 0; i < batch_size; i++) {
    const auto cur_label = label_data[i];
    if (cur_label == ignore_index) {
      out_data[i] = 0;
      continue;
    }
    PADDLE_ENFORCE_EQ(
        cur_label >= 0 && cur_label < n_classes,
        true,
        common::errors::InvalidArgument("label should not be out of bounds."));

    const auto cur_weight =
        weight_data ? weight_data[cur_label] : static_cast<T>(1);
    total_weight_val += cur_weight;
    output_val -= x_data[i * n_classes + cur_label] * cur_weight;
  }
  if (reduction == "mean" && total_weight_val != 0) {
    output_val /= total_weight_val;
  }
  *out_data = output_val;
  *total_weight_data = total_weight_val;
}

template <typename T>
static void nll_loss_2D(T* out_data,
                        T* total_weight_data,
                        const T* x_data,
                        const int64_t* label_data,
                        const T* weight_data,
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
            out_data[index] = 0;
            continue;
          }
          PADDLE_ENFORCE_EQ(cur_label >= 0 && cur_label < n_classes,
                            true,
                            common::errors::InvalidArgument(
                                "label should not be out of bounds."));
          const auto cur_weight =
              weight_data ? weight_data[cur_label] : static_cast<T>(1);
          out_data[index] = -x_data[i * sample_size + cur_label * map_size +
                                    h * in_dim3 + w] *
                            cur_weight;
        }
      }
    }
    return;
  }

  T output_val = 0;
  T total_weight_val = 0;

  for (int i = 0; i < batch_size; i++) {
    for (int h = 0; h < in_dim2; h++) {
      for (int w = 0; w < in_dim3; w++) {
        const auto index = i * map_size + h * in_dim3 + w;
        const auto cur_label = label_data[index];
        if (cur_label == ignore_index) {
          out_data[index] = 0;
          continue;
        }
        PADDLE_ENFORCE_EQ(cur_label >= 0 && cur_label < n_classes,
                          true,
                          common::errors::InvalidArgument(
                              "label should not be out of bounds."));
        const auto cur_weight =
            weight_data ? weight_data[cur_label] : static_cast<T>(1);
        total_weight_val += cur_weight;
        output_val -=
            x_data[i * sample_size + cur_label * map_size + h * in_dim3 + w] *
            cur_weight;
      }
    }
  }

  if (reduction == "mean" && total_weight_val != 0) {
    output_val /= total_weight_val;
  }
  *out_data = output_val;
  *total_weight_data = total_weight_val;
}

template <typename T, typename Context>
void NllLossRawKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& labels,
                      const paddle::optional<DenseTensor>& weight,
                      int64_t ignore_index,
                      const std::string& reduction,
                      DenseTensor* out,
                      DenseTensor* total_weight) {
  auto x_data = x.data<T>();
  auto label_data = labels.data<int64_t>();
  auto weight_data = weight.get_ptr() ? weight.get_ptr()->data<T>() : nullptr;
  auto out_data = dev_ctx.template Alloc<T>(out);
  auto total_weight_data = dev_ctx.template Alloc<T>(total_weight);
  *total_weight_data = 0;

  auto x_dims = x.dims();
  const auto batch_size = x_dims[0];
  const auto n_classes = x_dims[1];

  if (x_dims.size() == 2) {
    nll_loss_1D<T>(out_data,
                   total_weight_data,
                   x_data,
                   label_data,
                   weight_data,
                   batch_size,
                   n_classes,
                   reduction,
                   ignore_index);
  } else if (x_dims.size() == 4) {
    const auto in_dim2 = x_dims[2];
    const auto in_dim3 = x_dims[3];
    nll_loss_2D<T>(out_data,
                   total_weight_data,
                   x_data,
                   label_data,
                   weight_data,
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
    nll_loss, CPU, ALL_LAYOUT, phi::NllLossRawKernel, float, double) {}
