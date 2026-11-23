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
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/tile_grad_kernel.h"
#if defined(__NVCC__) || defined(__HIPCC__)
#include "paddle/phi/kernels/gpu/reduce.h"
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

  // A dim with repeat 1 has size 1, summing it is a no-op.
  int last_lead_sum = -1;
  for (int i = 0; i < num_unsqueezed; ++i) {
    if (repeat_times_data[i] != 1) {
      last_lead_sum = i;
    }
  }
  DenseTensor cur = out_grad;
  for (int i = 0; i <= last_lead_sum; ++i) {
    if (repeat_times_data[i] == 1) {
      continue;
    }
    if (i == last_lead_sum && grad_size.size() <= 1) {
      // Nothing is left to reduce, so the last sum writes x_grad directly.
      SumKernel<T, Context>(
          dev_ctx, cur, IntArray({0}), cur.dtype(), false, x_grad);
      x_grad->Resize(x_dims);
      return;
    }
    cur = Sum<T, Context>(dev_ctx, cur, IntArray({0}), cur.dtype(), false);
  }

  if (grad_size.size() <= 1) {
    dev_ctx.template Alloc<T>(x_grad);
    Copy(dev_ctx, cur, dev_ctx.GetPlace(), false, x_grad);
    // TensorCopy may change the dims of dx
    x_grad->Resize(x_dims);
    return;
  }

#if defined(__NVCC__) || defined(__HIPCC__)
  // One reduce over the whole interleaved view, as torch does. The view can
  // reach 2 * rank(x) = 14 dims, more than a DDim holds, so it is passed as a
  // shape vector.
  std::vector<int> reduce_dims;
  for (size_t d = 0; d < grad_size.size(); ++d) {
    if (is_repeat[d]) {
      reduce_dims.push_back(static_cast<int>(d));
    }
  }
  funcs::ReduceGpuKernel<T, T, kps::SumOps>(
      dev_ctx, cur, x_grad, reduce_dims, 1.0, grad_size);
#else
  // With every repeat != 1, grad_size reaches 2 * rank(x) = 14 dims, more
  // than a DDim holds. Then sum the repeat axes among the first
  // kMaxRank - 1 dims with the rest flattened, and go on with the result.
  // ponytail: CPU only; the extra pass rounds to T once more for
  // fp16/bf16/fp8, only ranks beyond DDim::kMaxRank take it.
  while (grad_size.size() > static_cast<size_t>(DDim::kMaxRank)) {
    const int head = DDim::kMaxRank - 1;
    std::vector<int64_t> shape(grad_size.begin(), grad_size.begin() + head);
    std::vector<int64_t> axes;
    int64_t kept = 1;
    int64_t rest = 1;
    for (int d = 0; d < head; ++d) {
      if (is_repeat[d]) {
        axes.push_back(d);
      } else {
        kept *= grad_size[d];
      }
    }
    for (size_t d = head; d < grad_size.size(); ++d) {
      rest *= grad_size[d];
    }
    shape.push_back(rest);
    cur.Resize(make_ddim(shape));
    cur = Sum<T, Context>(dev_ctx, cur, IntArray(axes), cur.dtype(), false);
    grad_size.erase(grad_size.begin(), grad_size.begin() + head);
    grad_size.insert(grad_size.begin(), kept);
    is_repeat.erase(is_repeat.begin(), is_repeat.begin() + head);
    is_repeat.insert(is_repeat.begin(), false);
  }

  std::vector<int64_t> sum_dims;
  std::vector<int64_t> kept_size;
  for (size_t d = 0; d < grad_size.size(); ++d) {
    if (is_repeat[d]) {
      sum_dims.push_back(d);
    } else {
      kept_size.push_back(grad_size[d]);
    }
  }
  cur.Resize(make_ddim(grad_size));
  x_grad->Resize(make_ddim(kept_size));
  SumKernel<T, Context>(
      dev_ctx, cur, IntArray(sum_dims), cur.dtype(), false, x_grad);
  x_grad->Resize(x_dims);
#endif
}

}  // namespace phi
