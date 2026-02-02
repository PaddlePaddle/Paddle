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

#include "paddle/phi/kernels/std_var_grad_kernel.h"

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/compare_kernel.h"
#include "paddle/phi/kernels/elementwise_divide_kernel.h"
#include "paddle/phi/kernels/elementwise_kernel.h"
#include "paddle/phi/kernels/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/reduce_mean_kernel.h"
#include "paddle/phi/kernels/scale_kernel.h"
#include "paddle/phi/kernels/unsqueeze_kernel.h"
#include "paddle/phi/kernels/where_kernel.h"

namespace phi {

template <typename T, typename Context>
void VarGradKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& out_grad,
                   const std::vector<int64_t>& axis,
                   bool keepdim,
                   bool unbiased,
                   double correction,
                   DenseTensor* x_grad) {
  int rank = x.dims().size();
  if (rank == 0 || axis.size() == 0) {
    const auto dof = static_cast<double>(x.numel()) - correction;
    DenseTensor x_mean = phi::Mean<T, Context>(dev_ctx, x, {}, true);
    if (dof <= 0) {
      // grad * at::where(x ==
      // x.mean(),std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity());
      DenseTensor cond;
      cond.Resize(x.dims());
      phi::EqualKernel<T, Context>(dev_ctx, x, x_mean, &cond);
      DenseTensor nan_tensor = phi::FullLike<T, Context>(
          dev_ctx, x, static_cast<T>(std::numeric_limits<double>::quiet_NaN()));
      DenseTensor inf_tensor = phi::FullLike<T, Context>(
          dev_ctx, x, static_cast<T>(std::numeric_limits<double>::infinity()));
      dev_ctx.template Alloc<T>(x_grad);
      phi::WhereKernel<T, Context>(
          dev_ctx, cond, nan_tensor, inf_tensor, x_grad);
    } else {
      // (2.0 / dof) * grad * (x - x.mean());
      DenseTensor diff = phi::Subtract<T, Context>(dev_ctx, x, x_mean);
      DenseTensor scale =
          phi::FullLike<T, Context>(dev_ctx, x, static_cast<T>(2.0 / dof));
      DenseTensor tmp = phi::Multiply<T, Context>(dev_ctx, scale, out_grad);
      dev_ctx.template Alloc<T>(x_grad);
      phi::MultiplyKernel<T, Context>(dev_ctx, tmp, diff, x_grad);
    }
    return;
  }

  std::vector<int64_t> axes64 = axis;
  for (auto& d : axes64)
    if (d < 0) d += rank;
  std::sort(axes64.begin(), axes64.end());
  axes64.erase(std::unique(axes64.begin(), axes64.end()), axes64.end());

  int64_t rnumel = 1;
  for (int d : axes64) {
    rnumel *= static_cast<int64_t>(x.dims()[d]);
  }
  double denom = static_cast<double>(rnumel) - correction;
  DenseTensor grad_expanded = out_grad;
  if (!keepdim && rank > 1) {
    IntArray unsq_axes(axes64);
    DenseTensor tmp;
    Unsqueeze<T, Context>(dev_ctx, out_grad, unsq_axes, &tmp, nullptr);
    grad_expanded = std::move(tmp);
  }

  // (2.0 / denom) * grad * (x - x.mean());
  DenseTensor x_mean = Mean<T, Context>(dev_ctx, x, axes64, /*keepdim=*/true);
  DenseTensor diff = Subtract<T, Context>(dev_ctx, x, x_mean);
  DenseTensor scale =
      phi::FullLike<T, Context>(dev_ctx, x, static_cast<T>(2.0 / denom));
  DenseTensor tmp = Multiply<T, Context>(dev_ctx, scale, grad_expanded);
  dev_ctx.template Alloc<T>(x_grad);
  MultiplyKernel<T, Context>(dev_ctx, tmp, diff, x_grad);
}

template <typename T, typename Context>
void StdGradKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& out,
                   const DenseTensor& out_grad,
                   const std::vector<int64_t>& axis,
                   bool keepdim,
                   bool unbiased,
                   double correction,
                   DenseTensor* x_grad) {
  // grad_var = (grad / (out * 2)).masked_fill_(out == 0, 0);
  DenseTensor two_tensor =
      phi::FullLike<T, Context>(dev_ctx, out, static_cast<T>(2.0));
  DenseTensor denom = Multiply<T, Context>(dev_ctx, out, two_tensor);
  DenseTensor div = Divide<T, Context>(dev_ctx, out_grad, denom);

  DenseTensor zero_tensor =
      phi::FullLike<T, Context>(dev_ctx, out, static_cast<T>(0.0));
  DenseTensor cond_zero;
  cond_zero.Resize(out.dims());
  EqualKernel<T, Context>(dev_ctx, out, zero_tensor, &cond_zero);
  DenseTensor grad_var;
  grad_var.Resize(out_grad.dims());
  WhereKernel<T, Context>(dev_ctx, cond_zero, zero_tensor, div, &grad_var);

  // call var_backward
  VarGradKernel<T, Context>(
      dev_ctx, x, grad_var, axis, keepdim, unbiased, correction, x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(var_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::VarGradKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}
PD_REGISTER_KERNEL(std_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::StdGradKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}
