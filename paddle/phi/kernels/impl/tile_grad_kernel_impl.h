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

#pragma once
#include <vector>

#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/tile_grad_kernel.h"
#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/phi/kernels/gpu/reduce.h"
#else
#include "paddle/phi/kernels/funcs/elementwise_grad_base.h"
#endif

namespace phi {

// Mirrors torch's repeat_backward: sum the leading unsqueezed dims one at a
// time, then view each tiled dim i as [repeat_i, x_dims[i]] and sum the
// repeat axes.
template <typename T, typename Context>
void TileGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& out_grad,
                    const IntArray& repeat_times,
                    DenseTensor* x_grad) {
  // x_grad->numel() may be not 0.
  if (out_grad.numel() == 0) {
    Full<T, Context>(dev_ctx, x_grad->dims(), 0, x_grad);
    return;
  }
  const auto x_dims = x.dims();
  const int x_rank = x_dims.size();
  auto repeat_times_data = repeat_times.GetData();
  if (repeat_times_data.size() < static_cast<size_t>(x_rank)) {
    repeat_times_data.insert(
        repeat_times_data.begin(), x_rank - repeat_times_data.size(), 1);
  }
  const int num_unsqueezed =
      static_cast<int>(repeat_times_data.size()) - x_rank;

  // Same as torch: grad_size holds [repeat_i, x_dims[i]] if repeat_i != 1,
  // else x_dims[i]; sum_dims are the positions of the repeat_i.
  std::vector<int64_t> grad_size;
  std::vector<int64_t> sum_dims;
  for (int i = 0; i < x_rank; ++i) {
    const int64_t repeat = repeat_times_data[i + num_unsqueezed];
    if (repeat != 1) {
      grad_size.push_back(repeat);
      sum_dims.push_back(static_cast<int64_t>(grad_size.size()) - 1);
    }
    grad_size.push_back(x_dims[i]);
  }

  DenseTensor cur = out_grad;
  for (int i = 0; i < num_unsqueezed; ++i) {
    // When nothing is left to reduce, the last sum writes x_grad directly.
    const bool last = i == num_unsqueezed - 1 && sum_dims.empty();
    const auto shape = common::vectorize<int64_t>(cur.dims());
    DenseTensor next;
    DenseTensor* out = last ? x_grad : &next;
    out->Resize(
        make_ddim(std::vector<int64_t>(shape.begin() + 1, shape.end())));
#if defined(__NVCC__) || defined(__HIPCC__)
    funcs::ReduceGpuKernel<T, T, kps::SumOps>(dev_ctx, cur, out, {0});
#else
    funcs::ReduceGradTermsCascade<T>(cur.data<T>(),
                                     shape,
                                     {0},
                                     dev_ctx.template Alloc<T>(out),
                                     out->numel());
#endif
    if (last) {
      return;
    }
    cur = next;
  }

  if (sum_dims.empty()) {
    dev_ctx.template Alloc<T>(x_grad);
    Copy(dev_ctx, cur, dev_ctx.GetPlace(), false, x_grad);
    // TensorCopy may change the dims of dx
    x_grad->Resize(x_dims);
    return;
  }

  // One reduce over the whole interleaved view, as torch does. The view can
  // reach 2 * rank(x) = 14 dims, more than a DDim holds, so it is passed as a
  // shape vector on both CPU and GPU.
#if defined(__NVCC__) || defined(__HIPCC__)
  funcs::ReduceSumGpuKernelWithShape<T, T>(
      dev_ctx, cur, x_grad, grad_size, sum_dims);
#else
  funcs::ReduceGradTermsCascade<T>(cur.data<T>(),
                                   grad_size,
                                   sum_dims,
                                   dev_ctx.template Alloc<T>(x_grad),
                                   x_grad->numel());
#endif
}

}  // namespace phi
