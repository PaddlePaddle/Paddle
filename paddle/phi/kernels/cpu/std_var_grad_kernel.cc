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
  if (x.numel() == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    return;
  }
  int rank = x.dims().size();
  // Match torch var_backward:
  //   dim=None / 0-D  -> n = numel
  //   dim=[]          -> n = 1  (falls through; empty axes product is 1)
  // Python maps axis=None to all dims, so empty axis here is a true dim=[].
  if (rank == 0) {
    const auto dof = static_cast<double>(x.numel()) - correction;
    DenseTensor x_mean = Mean<T, Context>(dev_ctx, x, {}, true);
    if (dof <= 0) {
      // grad * at::where(x ==
      // x.mean(),std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity());
      DenseTensor cond;
      cond.Resize(x.dims());
      EqualKernel<T, Context>(dev_ctx, x, x_mean, &cond);
      DenseTensor nan_tensor = FullLike<T, Context>(
          dev_ctx, x, static_cast<T>(std::numeric_limits<double>::quiet_NaN()));
      DenseTensor inf_tensor = FullLike<T, Context>(
          dev_ctx, x, static_cast<T>(std::numeric_limits<double>::infinity()));
      dev_ctx.template Alloc<T>(x_grad);
      WhereKernel<T, Context>(dev_ctx, cond, nan_tensor, inf_tensor, x_grad);
    } else {
      // (2.0 / dof) * grad * (x - x.mean());
      DenseTensor diff = Subtract<T, Context>(dev_ctx, x, x_mean);
      DenseTensor scale =
          FullLike<T, Context>(dev_ctx, x, static_cast<T>(2.0 / dof));
      DenseTensor tmp = Multiply<T, Context>(dev_ctx, scale, out_grad);
      dev_ctx.template Alloc<T>(x_grad);
      MultiplyKernel<T, Context>(dev_ctx, tmp, diff, x_grad);
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
  // `axes64` is empty only for dim=[], where out_grad already has the right
  // rank. GetUnsqueezeShape() copies in_dims verbatim for an empty axis list,
  // so this guard skips a no-op kernel call rather than changing the result.
  if (!keepdim && rank > 1 && !axes64.empty()) {
    IntArray unsq_axes(axes64);
    DenseTensor tmp;
    Unsqueeze<T, Context>(dev_ctx, out_grad, unsq_axes, &tmp, nullptr);
    grad_expanded = std::move(tmp);
  }

  // (2.0 / denom) * grad * (x - x.mean());
  DenseTensor x_mean = Mean<T, Context>(dev_ctx, x, axes64, /*keepdim=*/true);
  DenseTensor diff = Subtract<T, Context>(dev_ctx, x, x_mean);
  if (denom <= 0 && axes64.size() == static_cast<size_t>(rank)) {
    // Match torch's singular-degree-of-freedom rule.  The sign of x - mean is
    // intentionally discarded: non-constant inputs receive +Inf, while
    // constant inputs receive NaN.
    DenseTensor cond;
    cond.Resize(x.dims());
    // Compare x with the mean directly.  diff == 0 is not equivalent for
    // Inf inputs because Inf - Inf is NaN.
    EqualKernel<T, Context>(dev_ctx, x, x_mean, &cond);
    DenseTensor nan_tensor = FullLike<T, Context>(
        dev_ctx, x, static_cast<T>(std::numeric_limits<double>::quiet_NaN()));
    DenseTensor inf_tensor = FullLike<T, Context>(
        dev_ctx, x, static_cast<T>(std::numeric_limits<double>::infinity()));
    DenseTensor singular_grad;
    singular_grad.Resize(x.dims());
    WhereKernel<T, Context>(
        dev_ctx, cond, nan_tensor, inf_tensor, &singular_grad);
    dev_ctx.template Alloc<T>(x_grad);
    MultiplyKernel<T, Context>(dev_ctx, singular_grad, grad_expanded, x_grad);
    return;
  }
  DenseTensor scale =
      FullLike<T, Context>(dev_ctx, x, static_cast<T>(2.0 / denom));
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
  if (x.numel() == 0) {
    dev_ctx.template Alloc<T>(x_grad);
    return;
  }
  // grad_var = (grad / (out * 2)).masked_fill_(out == 0, 0);
  DenseTensor two_tensor =
      FullLike<T, Context>(dev_ctx, out, static_cast<T>(2.0));
  DenseTensor denom = Multiply<T, Context>(dev_ctx, out, two_tensor);
  DenseTensor div = Divide<T, Context>(dev_ctx, out_grad, denom);

  DenseTensor zero_tensor =
      FullLike<T, Context>(dev_ctx, out, static_cast<T>(0.0));
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

PD_REGISTER_KERNEL(
    var_grad, CPU, ALL_LAYOUT, phi::VarGradKernel, float, double) {}
PD_REGISTER_KERNEL(
    std_grad, CPU, ALL_LAYOUT, phi::StdGradKernel, float, double) {}
