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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {

template <typename T, typename Context>
void HistogramKernel(const Context& dev_ctx,
                     const DenseTensor& input,
                     const DenseTensor& bins,
                     int min,
                     int max,
                     DenseTensor* output) {
  const T* input_data = input.data<T>();
  auto input_numel = input.numel();

  if (input_data == nullptr) return;

  const T* bins_data = bins.data<T>();

  auto bins_numel = bins.numel();
  if (bins_numel == 1) {
    int64_t nbins = static_cast<int64_t>(bins_data[0]);
    PADDLE_ENFORCE_GE(
        nbins,
        1,
        phi::errors::InvalidArgument(
            "When bins is an int, it should be greater than or equal to 1, "
            "but the value received is %d",
            nbins));

    output->Resize(phi::make_ddim({static_cast<int64_t>(nbins)}));
    int64_t* out_data = output->mutable_data<int64_t>(dev_ctx.GetPlace());
    phi::funcs::SetConstant<Context, int64_t>()(
        dev_ctx, output, static_cast<int64_t>(0));

    auto& minval = min;
    auto& maxval = max;

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

    PADDLE_ENFORCE_EQ(
        (std::isinf(static_cast<float>(output_min)) ||
         std::isnan(static_cast<float>(output_max)) ||
         std::isinf(static_cast<float>(output_max)) ||
         std::isnan(static_cast<float>(output_min))),
        false,
        phi::errors::OutOfRange("range of min, max is not finite"));
    PADDLE_ENFORCE_GE(
        output_max,
        output_min,
        phi::errors::InvalidArgument(
            "max must be larger or equal to min. If min is equal to max, "
            "the minimum and maximum values of the data are used. "
            "But received max is %d, min is %d",
            maxval,
            minval));

    for (int64_t i = 0; i < input_numel; i++) {
      if (input_data[i] >= output_min && input_data[i] <= output_max) {
        const int64_t bin = (int64_t)((input_data[i] - output_min) * nbins /
                                      (output_max - output_min));
        out_data[std::min(bin, nbins - 1)] += 1;
      }
    }
  } else {
    for (int64_t i = 0; i < bins_numel - 1; i++) {
      PADDLE_ENFORCE_GT(bins_data[i + 1],
                        bins_data[i],
                        phi::errors::InvalidArgument(
                            "When bins is a list or a Tensor, it should "
                            "increase monotonically, "
                            "but bins_%d is less than or equal to bins_%d",
                            i + 1,
                            i));
    }

    int64_t* out_data = output->mutable_data<int64_t>(dev_ctx.GetPlace());
    phi::funcs::SetConstant<Context, int64_t>()(
        dev_ctx, output, static_cast<int64_t>(0));

    auto& output_min = bins_data[0];
    auto& output_max = bins_data[bins_numel - 1];
    for (int64_t i = 0; i < input_numel; i++) {
      auto input_data_i = input_data[i];
      if (input_data_i >= output_min && input_data_i <= output_max) {
        int64_t l = 0;
        int64_t r = bins_numel;
        while (l < r) {
          int mid = l + (r - l) / 2;
          if (bins_data[mid] <= input_data_i) {
            l = mid + 1;
          } else {
            r = mid;
          }
        }
        out_data[std::min(l - 1, output->numel() - 1)] += 1;
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
