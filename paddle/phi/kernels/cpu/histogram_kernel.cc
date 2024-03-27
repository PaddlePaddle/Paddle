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

#include "paddle/phi/kernels/histogram_kernel.h"
#include <cstdint>

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/backends/device_ext.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/int_array.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/utils/optional.h"

namespace phi {
template <typename T, typename Context>
void HistogramKernel(const Context& dev_ctx,
                     const DenseTensor& input,
                     const paddle::optional<DenseTensor>& weight,
                     bool density,
                     int64_t bins,
                     int min,
                     int max,
                     DenseTensor* output) {
  auto& nbins = bins;
  auto& minval = min;
  auto& maxval = max;

  const T* input_data = input.data<T>();
  auto weight_data = weight.get_ptr() ? weight.get_ptr()->data<T>() : nullptr;
  auto input_numel = input.numel();

  if (input_data == nullptr) return;

  T output_min = static_cast<T>(minval);
  T output_max = static_cast<T>(maxval);
  if (output_min == output_max) {
    output_min = *std::min_element(input_data, input_data + input_numel);
    output_max = *std::max_element(input_data, input_data + input_numel);
  }
  if (output_min == output_max) {
    output_min = output_min - 1;
    output_max = output_max + 1;
  }

  PADDLE_ENFORCE_EQ((std::isinf(static_cast<float>(output_min)) ||
                     std::isnan(static_cast<float>(output_max)) ||
                     std::isinf(static_cast<float>(output_min)) ||
                     std::isnan(static_cast<float>(output_max))),
                    false,
                    phi::errors::OutOfRange("range of min, max is not finite"));
  PADDLE_ENFORCE_GE(
      output_max,
      output_min,
      phi::errors::InvalidArgument(
          "max must be larger or equal to min. If min and max are both zero, "
          "the minimum and maximum values of the data are used. "
          "But received max is %d, min is %d",
          maxval,
          minval));

  if (density || weight_data) {
    float* out_data = dev_ctx.template Alloc<float>(output);
    phi::funcs::SetConstant<Context, float>()(
        dev_ctx, output, static_cast<float>(0));
    for (int64_t i = 0; i < input_numel; i++) {
      if (input_data[i] >= output_min && input_data[i] <= output_max) {
        const int64_t bin = (int64_t)((input_data[i] - output_min) * nbins /
                                      (output_max - output_min));
        out_data[std::min(bin, nbins - 1)] +=
            weight_data ? static_cast<float>(weight_data[i]) : 1;
      }
    }
    if (density) {
      DenseTensor sum = phi::Sum<float, Context>(
          dev_ctx, *output, phi::IntArray({0}), phi::DataType::FLOAT32, false);
      float* sum_data = sum.data<float>();
      float gap = static_cast<float>(nbins) /
                  static_cast<float>((output_max - output_min)) / *sum_data;
      for (int64_t i = 0; i < nbins; i++) {
        out_data[i] *= gap;
      }
    }
  } else {
    int64_t* out_data = dev_ctx.template Alloc<int64_t>(output);
    phi::funcs::SetConstant<Context, int64_t>()(
        dev_ctx, output, static_cast<int64_t>(0));
    for (int64_t i = 0; i < input_numel; i++) {
      if (input_data[i] >= output_min && input_data[i] <= output_max) {
        const int64_t bin = (int64_t)((input_data[i] - output_min) * nbins /
                                      (output_max - output_min));
        out_data[std::min(bin, nbins - 1)] += 1;
      }
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(histogram,
                   CPU,
                   ALL_LAYOUT,
                   phi::HistogramKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
