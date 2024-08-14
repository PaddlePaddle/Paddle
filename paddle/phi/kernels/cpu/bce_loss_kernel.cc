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

#include "paddle/phi/kernels/bce_loss_kernel.h"

#include <algorithm>  // for max

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math.h"

namespace phi {

template <typename T, typename Context>
void BCELossKernel(const Context& dev_ctx,
                   const DenseTensor& input,
                   const DenseTensor& label,
                   DenseTensor* out) {
  auto x_data = input.data<T>();
  auto label_data = label.data<T>();
  auto out_data = dev_ctx.template Alloc<T>(out);
  auto x_numel = input.numel();

  // out = -(label * ln(x) + (1 - label) * ln(1 - x)) = (label - 1) * ln(1 -
  // x) - label * ln(x)
  for (int64_t i = 0; i < x_numel; ++i) {
    PADDLE_ENFORCE_GE(
        x_data[i],
        static_cast<T>(0),
        common::errors::InvalidArgument(
            "Illegal input, input must be greater than  or equal to 0"));
    PADDLE_ENFORCE_LE(
        x_data[i],
        static_cast<T>(1),
        common::errors::InvalidArgument(
            "Illegal input, input must be less than or equal to 1"));
    out_data[i] =
        (label_data[i] - static_cast<T>(1)) *
            std::max(phi::funcs::real_log(static_cast<T>(1) - x_data[i]),
                     (T)(-100)) -
        label_data[i] * std::max(phi::funcs::real_log(x_data[i]), (T)(-100));
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    bce_loss, CPU, ALL_LAYOUT, phi::BCELossKernel, float, double) {}
