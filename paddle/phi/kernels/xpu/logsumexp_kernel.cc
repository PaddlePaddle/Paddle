// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/phi/kernels/logsumexp_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/kernels/reduce_max_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

template <typename T, typename Context>
void LogsumexpKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const std::vector<int>& axis,
                     bool keepdim,
                     bool reduce_all,
                     DenseTensor* out) {
  auto xdim = x.dims();
  for (int i = 0; i < xdim.size(); i++)
    PADDLE_ENFORCE_LT(0,
                      xdim[i],
                      errors::InvalidArgument(
                          "The dims of Input(X) should be greater than 0."));

  reduce_all = recompute_reduce_all(x, axis, reduce_all);
  std::vector<int64_t> outdim_vec, keeped_outdim_vec;
  std::vector<int> axis_vec;
  int64_t compute_size = 1, other_size = 1;
  for (auto i : axis) {
    auto v = i >= 0 ? i : i + xdim.size();
    axis_vec.push_back(v);
  }
  if (axis.size() == 0 || reduce_all) {
    axis_vec.clear();
    for (int i = 0; i < xdim.size(); i++) {
      axis_vec.push_back(i);
    }
  }
  for (int i = 0; i < xdim.size(); i++) {
    bool flag = false;
    for (auto v : axis_vec) {
      if (v == i) {
        flag = true;
        break;
      }
    }
    if (flag) {
      compute_size *= xdim[i];
      keeped_outdim_vec.push_back(1);
      if (keepdim) outdim_vec.push_back(1);
    } else {
      other_size *= xdim[i];
      outdim_vec.push_back(xdim[i]);
      keeped_outdim_vec.push_back(xdim[i]);
    }
  }
  auto outdim = common::make_ddim(outdim_vec);
  auto keeped_outdim = common::make_ddim(keeped_outdim_vec);

  // The XPU logsumexp api does not use xmax to normalize its input, so we
  // fallback to the non fusion impl currently.
  DenseTensor max_x;
  max_x.Resize(keeped_outdim);
  MaxKernel<T, Context>(dev_ctx, x, axis_vec, true, &max_x);

  DenseTensor temp_x = Subtract<T, Context>(dev_ctx, x, max_x);
  ExpKernel<T, Context>(dev_ctx, temp_x, &temp_x);
  SumKernel<T, Context>(dev_ctx, temp_x, axis_vec, x.dtype(), keepdim, out);
  LogKernel<T, Context>(dev_ctx, *out, out);
  max_x.Resize(outdim);
  out->Resize(outdim);
  AddKernel<T, Context>(dev_ctx, *out, max_x, out);
}
}  // namespace phi

PD_REGISTER_KERNEL(logsumexp,
                   XPU,
                   ALL_LAYOUT,
                   phi::LogsumexpKernel,
                   float,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
