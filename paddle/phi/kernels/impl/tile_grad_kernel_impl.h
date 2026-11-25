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

  // grad_size interleaves [repeat_i, x_dims[i]] for every repeat_i != 1.
  // Untiled dims are merged into the preceding kept dim, which the reduce
  // kernel would collapse anyway, so the reduction itself is unchanged.
  std::vector<int64_t> grad_size;
  std::vector<bool> is_repeat;
  for (int i = 0; i < x_rank; ++i) {
    const int64_t repeat = repeat_times_data[i + num_unsqueezed];
    if (repeat != 1) {
      grad_size.push_back(repeat);
      is_repeat.push_back(true);
    } else if (!grad_size.empty()) {
      grad_size.back() *= x_dims[i];
      continue;
    }
    grad_size.push_back(x_dims[i]);
    is_repeat.push_back(false);
  }

  // A leading dim with repeat 1 has size 1: drop it with a reshape instead of
  // a sum, so that each sum over dim 0 below lands on a real repeat dim.
  DenseTensor cur = out_grad;
  std::vector<int64_t> cur_shape;
  int num_lead_sums = 0;
  for (int i = 0; i < cur.dims().size(); ++i) {
    if (i < num_unsqueezed && repeat_times_data[i] == 1) {
      continue;
    }
    num_lead_sums += i < num_unsqueezed;
    cur_shape.push_back(cur.dims()[i]);
  }
  cur.Resize(make_ddim(cur_shape));
  for (int i = 0; i < num_lead_sums; ++i) {
    // When nothing is left to reduce, the last sum writes x_grad directly.
    const bool last = i == num_lead_sums - 1 && grad_size.size() <= 1;
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
    if (!last) {
      cur = next;
      continue;
    }
    x_grad->Resize(x_dims);
    return;
  }

  if (grad_size.size() <= 1) {
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
  std::vector<int> reduce_dims;
  for (size_t d = 0; d < grad_size.size(); ++d) {
    if (is_repeat[d]) {
      reduce_dims.push_back(static_cast<int>(d));
    }
  }
  funcs::ReduceSumGpuKernelWithShape<T, T>(
      dev_ctx, cur, x_grad, grad_size, reduce_dims);
#else
  std::vector<int64_t> sum_dims;
  for (size_t d = 0; d < grad_size.size(); ++d) {
    if (is_repeat[d]) {
      sum_dims.push_back(static_cast<int64_t>(d));
    }
  }
  funcs::ReduceGradTermsCascade<T>(cur.data<T>(),
                                   grad_size,
                                   sum_dims,
                                   dev_ctx.template Alloc<T>(x_grad),
                                   x_grad->numel());
#endif
}

}  // namespace phi
