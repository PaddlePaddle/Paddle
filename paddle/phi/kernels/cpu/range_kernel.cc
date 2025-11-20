/* Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/range_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/range_function.h"

namespace phi {

template <typename T, typename Context>
void RangeFunc(const Context& dev_ctx,
               const T& start_value,
               const T& end_value,
               const T& step_value,
               DenseTensor* out) {
  int64_t size =
      static_cast<int64_t>((end_value - start_value) / step_value + 1);
  out->Resize(common::make_ddim({size}));
  T* out_data = dev_ctx.template Alloc<T>(out);
  if (size == 0) {
    return;
  }
  T value = start_value;
  for (int64_t i = 0; i < size; ++i) {
    out_data[i] = value;
    value += step_value;
  }
}

template <typename T, typename Context>
void RangeTensorKernel(const Context& dev_ctx,
                       const DenseTensor& start,
                       const DenseTensor& end,
                       const DenseTensor& step,
                       DenseTensor* out) {
  T start_value = start.data<T>()[0];
  T end_value = end.data<T>()[0];
  T step_value = step.data<T>()[0];
  if (step_value == static_cast<T>(0)) {
    PADDLE_THROW(errors::InvalidArgument("step must be nonzero."));
  }
  RangeFunc<T, Context>(dev_ctx, start_value, end_value, step_value, out);
}

template <typename T, typename Context>
void RangeKernel(const Context& dev_ctx,
                 const Scalar& start,
                 const Scalar& end,
                 const Scalar& step,
                 DenseTensor* out) {
  T start_value = start.to<T>();
  T end_value = end.to<T>();
  T step_value = step.to<T>();
  if constexpr (std::is_floating_point_v<T>) {
    if (std::isnan(end_value)) {
      PADDLE_THROW(phi::errors::InvalidArgument(
          "The end value of range cannot be NaN. Please check your input."));
    }
  }
  RangeFunc<T, Context>(dev_ctx, start_value, end_value, step_value, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(range_tensor,
                   CPU,
                   ALL_LAYOUT,
                   phi::RangeTensorKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
PD_REGISTER_KERNEL(
    range, CPU, ALL_LAYOUT, phi::RangeKernel, float, double, int, int64_t) {}
