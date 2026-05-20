// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/std_var_kernel.h"

#include "paddle/phi/backends/all_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/reduce_mean_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/scale_kernel.h"

namespace phi {

template <typename T, typename Context>
void VarKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& axis,
               bool keepdim,
               bool unbiased,
               double correction,
               DenseTensor* out) {
  if (x.numel() == 0) {
    Full<T, Context>(dev_ctx, out->dims(), static_cast<T>(NAN), out);
    return;
  }
  // 1. Mean
  // Use keepdim=true for broadcasting in subtraction
  DenseTensor mean_val = Mean<T, Context>(dev_ctx, x, axis, true);

  // 2. Subtract: x - mean
  DenseTensor sub_res = Subtract<T, Context>(dev_ctx, x, mean_val);

  // 3. Square: (x - mean)^2
  DenseTensor sq_res = Multiply<T, Context>(dev_ctx, sub_res, sub_res);

  // 4. Sum: Sum((x - mean)^2)
  DenseTensor sum = Sum<T, Context>(dev_ctx, sq_res, axis, x.dtype(), keepdim);

  // 5. Divide by (N - correction)
  double n = static_cast<double>(x.numel()) / static_cast<double>(out->numel());
  double divisor = 0;
  if (n - correction >= 0) {
    divisor = 1.0 / (n - correction);
  }

  DenseTensor scale_val =
      FullLike<T, Context>(dev_ctx, *out, static_cast<T>(divisor));
  MultiplyKernel<T, Context>(dev_ctx, sum, scale_val, out);
}

template <typename T, typename Context>
void StdKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<int64_t>& axis,
               bool keepdim,
               bool unbiased,
               double correction,
               DenseTensor* out) {
  if (x.numel() == 0) {
    Full<T, Context>(dev_ctx, out->dims(), static_cast<T>(NAN), out);
    return;
  }
  VarKernel<T, Context>(dev_ctx, x, axis, keepdim, unbiased, correction, out);
  SqrtKernel<T, Context>(dev_ctx, *out, out);
}

}  // namespace phi
PD_REGISTER_KERNEL(var, CPU, ALL_LAYOUT, phi::VarKernel, float, double) {}
PD_REGISTER_KERNEL(std, CPU, ALL_LAYOUT, phi::StdKernel, float, double) {}
