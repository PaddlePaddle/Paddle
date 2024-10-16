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

#include "paddle/phi/kernels/dist_grad_kernel.h"

#include <tuple>
#include <vector>
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/kernels/p_norm_grad_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/scale_kernel.h"

namespace phi {

std::pair<std::vector<int64_t>, std::vector<int64_t>> GetReduceDims(
    const DDim& src_dim, const DDim& dst_dim) {
  std::vector<int64_t> reduce_dims, new_dims;
  auto pre_dims = src_dim.size() - dst_dim.size();
  for (auto i = 0; i < pre_dims; ++i) {
    reduce_dims.push_back(i);
  }

  for (auto i = pre_dims; i < src_dim.size(); ++i) {
    if (dst_dim[i - pre_dims] == 1 && src_dim[i] != 1) {
      reduce_dims.push_back(i);
    } else {
      new_dims.push_back(dst_dim[i - pre_dims]);
    }
  }
  return {reduce_dims, new_dims};
}

template <typename T, typename Context>
void DistGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    const DenseTensor& out,
                    const DenseTensor& out_grad,
                    float p,
                    DenseTensor* x_grad,
                    DenseTensor* y_grad) {
  if ((!x_grad) && (!y_grad)) {
    return;
  }

  auto t = Subtract<T, Context>(dev_ctx, x, y);
  DenseTensor x_grad_tmp;
  x_grad_tmp.Resize(t.dims());
  DenseTensor y_grad_tmp;
  y_grad_tmp.Resize(t.dims());
  PNormGradKernel<T, Context>(
      dev_ctx, t, out, out_grad, p, -1, 1e-12, false, true, &x_grad_tmp);

  if (x_grad) {
    // do reduce, the implementation of cpu SumKernel has bug, it changes
    // the dims of output internally, so we Resize x/y_grad twice.
    auto res_x = GetReduceDims(x_grad_tmp.dims(), x.dims());
    if (!std::get<0>(res_x).empty()) {
      x_grad->Resize(common::make_ddim(std::get<1>(res_x)));
      SumKernel<T, Context>(
          dev_ctx, x_grad_tmp, std::get<0>(res_x), x.dtype(), false, x_grad);
      x_grad->Resize(x.dims());
    } else {
      x_grad->ShareBufferWith(x_grad_tmp);
    }
  }

  if (y_grad) {
    ScaleKernel<T, Context>(dev_ctx, x_grad_tmp, -1.0, 0.0, false, &y_grad_tmp);
    auto res_y = GetReduceDims(y_grad_tmp.dims(), y.dims());
    if (!std::get<0>(res_y).empty()) {
      y_grad->Resize(common::make_ddim(std::get<1>(res_y)));
      SumKernel<T, Context>(
          dev_ctx, y_grad_tmp, std::get<0>(res_y), y.dtype(), false, y_grad);
      y_grad->Resize(y.dims());
    } else {
      y_grad->ShareBufferWith(y_grad_tmp);
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    dist_grad, CPU, ALL_LAYOUT, phi::DistGradKernel, float, double) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(dist_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::DistGradKernel,
                   float,
                   double,
                   phi::dtype::bfloat16,
                   phi::dtype::float16) {}
#endif
