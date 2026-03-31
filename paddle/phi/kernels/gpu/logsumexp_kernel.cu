// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/reduce_max_kernel.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

template <typename T>
struct ComputeType {
  using type = T;
};

template <>
struct ComputeType<phi::float16> {
  using type = float;
};

template <>
struct ComputeType<phi::bfloat16> {
  using type = float;
};

// Fused log(x) + y kernel that computes log in native precision.
// PyTorch's log kernel uses ::log(float) directly, while Paddle's LogKernel
// promotes float to double before computing log. This causes 1 ULP differences.
// This functor matches PyTorch's behavior by computing log in native precision.
// For half types (float16/bfloat16), compute log in float then round back to T
// before adding max, matching PyTorch's separate log_() then add_() pattern
// (result.log_().add_(maxes_squeezed) in ReduceOps.cpp line 1499).
template <typename T>
struct LogAddFunctor {
  using CT = typename ComputeType<T>::type;
  __device__ __forceinline__ T operator()(T sum_val, T max_val) const {
    T log_val = static_cast<T>(::log(static_cast<CT>(sum_val)));
    return log_val + max_val;
  }
};

template <typename T, typename Context>
void LogsumexpKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const std::vector<int>& axis_in,
                     bool keepdim,
                     bool reduce_all,
                     DenseTensor* out) {
  if (x.numel() == 0) {
    phi::Full<T, Context>(
        dev_ctx, phi::IntArray(common::vectorize(out->dims())), -INFINITY, out);
    return;
  }
  std::vector<int64_t> axis;
  axis.reserve(axis_in.size());
  std::for_each(axis_in.begin(), axis_in.end(), [&axis](const int& t) {
    axis.push_back(static_cast<int64_t>(t));
  });
  auto xdim = x.dims();
  for (size_t i = 0; i < xdim.size(); i++)
    PADDLE_ENFORCE_LT(0,
                      xdim[i],
                      errors::InvalidArgument(
                          "The dims of Input(X) should be greater than 0."));

  reduce_all = recompute_reduce_all(x, axis, reduce_all);
  std::vector<int64_t> outdim_vec, keeped_outdim_vec;
  std::vector<int> axis_vec;
  for (auto i : axis) {
    auto v = i >= 0 ? i : i + xdim.size();
    axis_vec.push_back(v);
  }
  if (axis.size() == 0 || reduce_all) {
    axis_vec.clear();
    for (size_t i = 0; i < xdim.size(); i++) {
      axis_vec.push_back(i);
    }
  }
  for (size_t i = 0; i < xdim.size(); i++) {
    bool is_reduce_dim = false;
    for (auto v : axis_vec) {
      if (v == i) {
        is_reduce_dim = true;
        break;
      }
    }
    if (is_reduce_dim) {
      keeped_outdim_vec.push_back(1);
      if (keepdim) outdim_vec.push_back(1);
    } else {
      outdim_vec.push_back(xdim[i]);
      keeped_outdim_vec.push_back(xdim[i]);
    }
  }

  auto outdim = common::make_ddim(outdim_vec);
  auto keeped_outdim = common::make_ddim(keeped_outdim_vec);

  // PyTorch-compatible logsumexp using composite operations:
  //   max → subtract → exp → sum → fused log+add
  // The log step uses native-precision ::log() to match PyTorch,
  // instead of Paddle's LogKernel which promotes float→double→float.

  // Step 1: max along reduction dims (keepdim=true for broadcasting)
  DenseTensor max_x;
  max_x.Resize(keeped_outdim);
  dev_ctx.template Alloc<T>(&max_x);
  phi::MaxKernel<T, Context>(dev_ctx, x, axis_vec, true, &max_x);

  // Step 2: subtract max from input (broadcasts keeped_outdim → x.dims)
  DenseTensor temp_x = Subtract<T, Context>(dev_ctx, x, max_x);

  // Step 3: exp(x - max)
  phi::ExpKernel<T, Context>(dev_ctx, temp_x, &temp_x);

  // Step 4: sum along reduction dims
  phi::SumKernel<T, Context>(
      dev_ctx, temp_x, axis_vec, x.dtype(), keepdim, out);

  // Step 5+6: fused log(sum) + max — using native-precision log
  max_x.Resize(outdim);
  out->Resize(outdim);
  std::vector<const DenseTensor*> ins = {out, &max_x};
  std::vector<DenseTensor*> outs = {out};
  funcs::BroadcastKernel<T>(
      dev_ctx, ins, &outs, LogAddFunctor<T>(), /*axis=*/-1);
}

}  // namespace phi

PD_REGISTER_KERNEL(logsumexp,
                   GPU,
                   ALL_LAYOUT,
                   phi::LogsumexpKernel,
                   float,
                   double,
                   phi::float16,
                   phi::bfloat16) {}
