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
                     int64_t bins,
                     int min,
                     int max,
                     DenseTensor* output) {
  auto& nbins = bins;
  auto& minval = min;
  auto& maxval = max;

  const T* input_data = input.data<T>();
  auto input_numel = input.numel();

  int64_t* out_data = dev_ctx.template Alloc<int64_t>(output);
  phi::funcs::SetConstant<Context, int64_t>()(
      dev_ctx, output, static_cast<int64_t>(0));

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

  // check if out of range
  double range =
      static_cast<double>(output_max) - static_cast<double>(output_min);
  PADDLE_ENFORCE_LT(
      range,
      static_cast<double>(std::numeric_limits<T>::max()),
      phi::errors::InvalidArgument(
          "The range of max - min is out of range for target type, "
          "current kernel type is %s, the range should less than %f "
          "but now min is %f, max is %f.",
          typeid(T).name(),
          std::numeric_limits<T>::max(),
          output_min,
          output_max));

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

  for (int64_t i = 0; i < input_numel; i++) {
    if (input_data[i] >= output_min && input_data[i] <= output_max) {
      const int64_t bin = (int64_t)((input_data[i] - output_min) * nbins /
                                    (output_max - output_min));
      out_data[std::min(bin, nbins - 1)] += 1;
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
                   int64_t) {
  kernel->OutputAt(0).SetDataType(paddle::DataType::INT64);
}
