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

#include <vector>
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/kernels/p_norm_grad_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/scale_kernel.h"

namespace phi {

std::vector<int64_t> GetReduceDims(const DDim& src_dim, const DDim& dst_dim) {
  std::vector<int64_t> reduce_dims;
  auto pre_dims = src_dim.size() - dst_dim.size();
  for (auto i = 0; i < pre_dims; ++i) {
    reduce_dims.push_back(i);
  }

  for (auto i = pre_dims; i < src_dim.size(); ++i) {
    if (dst_dim[i - pre_dims] == 1 && src_dim[i] != 1) {
      reduce_dims.push_back(i);
    }
  }
  return reduce_dims;
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
  auto t = Subtract<T, Context>(dev_ctx, x, y);
  DenseTensor x_grad_tmp;
  x_grad_tmp.Resize(t.dims());
  DenseTensor y_grad_tmp;
  y_grad_tmp.Resize(t.dims());
  PNormGradKernel<T, Context>(
      dev_ctx, t, out, out_grad, p, -1, 1e-12, false, true, &x_grad_tmp);
  ScaleKernel<T, Context>(dev_ctx, x_grad_tmp, -1.0, 0.0, false, &y_grad_tmp);
  // do reduce
  auto reduce_dims_x = GetReduceDims(x_grad_tmp.dims(), x.dims());
  if (!reduce_dims_x.empty()) {
    SumKernel<T, Context>(
        dev_ctx, x_grad_tmp, reduce_dims_x, x.dtype(), false, x_grad);
  } else {
    x_grad->ShareBufferWith(x_grad_tmp);
  }
  auto reduce_dims_y = GetReduceDims(y_grad_tmp.dims(), y.dims());
  if (!reduce_dims_y.empty()) {
    SumKernel<T, Context>(
        dev_ctx, y_grad_tmp, reduce_dims_y, y.dtype(), false, y_grad);
  } else {
    y_grad->ShareBufferWith(y_grad_tmp);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(
    dist_grad, CPU, ALL_LAYOUT, phi::DistGradKernel, float, double) {}

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
PD_REGISTER_KERNEL(
    dist_grad, GPU, ALL_LAYOUT, phi::DistGradKernel, float, double) {}
#endif
