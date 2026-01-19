// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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
               const IntArray& axis,
               bool keepdim,
               bool unbiased,
               double correction,
               DenseTensor* out) {
  // 1. Mean
  // Use keepdim=true for broadcasting in subtraction
  DenseTensor mean_val = phi::Mean<T, Context>(dev_ctx, x, axis, true);

  // 2. Subtract: x - mean
  DenseTensor sub_res = phi::Subtract<T, Context>(dev_ctx, x, mean_val);

  // 3. Square: (x - mean)^2
  DenseTensor sq_res = phi::Multiply<T, Context>(dev_ctx, sub_res, sub_res);

  // 4. Sum: Sum((x - mean)^2)
  DenseTensor sum =
      phi::Sum<float, Context>(dev_ctx, sq_res, axis, x.dtype(), keepdim);

  // 5. Divide by (N - correction)
  auto x_numel = x.numel();
  auto out_numel = out->numel();
  if (out_numel == 0) return;

  double n = static_cast<double>(x_numel) / static_cast<double>(out_numel);
  double divisor = n - correction;
  if (divisor < 0) divisor = 0;

  DenseTensor scale_val =
      phi::FullLike<T, Context>(dev_ctx, *out, static_cast<T>(1.0 / divisor));
  phi::MultiplyKernel<T, Context>(dev_ctx, sum, scale_val, out);
}

template <typename T, typename Context>
void StdKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const IntArray& axis,
               bool keepdim,
               bool unbiased,
               double correction,
               DenseTensor* out) {
  VarKernel<T, Context>(dev_ctx, x, axis, keepdim, unbiased, 1, out);
  SqrtKernel<T, Context>(dev_ctx, *out, out);
}

}  // namespace phi
PD_REGISTER_KERNEL(var, CPU, ALL_LAYOUT, phi::VarKernel, float, double) {}
PD_REGISTER_KERNEL(std, CPU, ALL_LAYOUT, phi::StdKernel, float, double) {}
