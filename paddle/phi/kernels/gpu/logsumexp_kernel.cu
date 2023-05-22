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

#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/elementwise_subtract_kernel.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/gpu/reduce.h"

namespace phi {

template <typename T>
struct LogCUDAFunctor {
  HOSTDEVICE inline T operator()(const T x) const { return std::log(x); }
};

template <>
struct LogCUDAFunctor<float16> {
  HOSTDEVICE inline float16 operator()(const float16 x) const {
    auto x_ = static_cast<float>(x);
    return static_cast<float16>(std::log(x_));
  }
};

template <>
struct LogCUDAFunctor<bfloat16> {
  HOSTDEVICE inline bfloat16 operator()(const bfloat16 x) const {
    auto x_ = static_cast<float>(x);
    return static_cast<bfloat16>(std::log(x_));
  }
};

template <typename T, typename Context>
void LogsumexpKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const std::vector<int64_t>& axis,
                     bool keepdim,
                     bool reduce_all,
                     DenseTensor* out) {
  auto* in_x = &x;
  auto* out_y = out;
  auto xdim = in_x->dims();
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
    for (size_t i = 0; i < xdim.size(); i++) {
      axis_vec.push_back(i);
    }
  }
  for (size_t i = 0; i < xdim.size(); i++) {
    bool flag = false;
    for (auto v : axis_vec) {
      if (v == i) {
        flag = true;
        break;
      }
    }
    if (flag) {
      keeped_outdim_vec.push_back(1);
      if (keepdim) outdim_vec.push_back(1);
    } else {
      outdim_vec.push_back(xdim[i]);
      keeped_outdim_vec.push_back(xdim[i]);
    }
  }

  auto outdim = phi::make_ddim(outdim_vec);
  auto keeped_outdim = phi::make_ddim(keeped_outdim_vec);
  out->Resize(outdim);
  dev_ctx.template Alloc<T>(out_y);

  DenseTensor max_x;
  max_x.Resize(outdim);
  dev_ctx.template Alloc<T>(&max_x);

  phi::funcs::ReduceKernel<T, T, kps::MaxFunctor, kps::IdentityFunctor<T>>(
      dev_ctx, *in_x, &max_x, kps::IdentityFunctor<T>(), axis_vec);

  max_x.Resize(keeped_outdim);
  DenseTensor temp_x = Subtract<T, Context>(dev_ctx, *in_x, max_x);
  phi::funcs::ReduceKernel<T, T, kps::AddFunctor, kps::ExpFunctor<T>>(
      dev_ctx, temp_x, out_y, kps::ExpFunctor<T>(), axis_vec);

  const std::vector<const DenseTensor*> inputs = {out_y};
  std::vector<DenseTensor*> outputs = {&temp_x};
  phi::funcs::ElementwiseKernel<T>(
      dev_ctx, inputs, &outputs, LogCUDAFunctor<T>());
  temp_x.Resize(outdim);
  out->Resize(outdim);
  phi::AddKernel<T, Context>(dev_ctx, temp_x, max_x, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(logsumexp,
                   GPU,
                   ALL_LAYOUT,
                   phi::LogsumexpKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
