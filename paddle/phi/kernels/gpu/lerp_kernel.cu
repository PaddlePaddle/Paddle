// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/lerp_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/expand_kernel.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"

namespace phi {

template <typename T>
struct LerpElementWiseDirectCUDAFunctor {
  HOSTDEVICE inline T operator()(const T x, const T y, const T weight) const {
    return x + weight * (y - x);
  }
};

template <typename T>
struct LerpScalarDirectCUDAFunctor {
  const T *weight_;

  HOSTDEVICE inline LerpScalarDirectCUDAFunctor(const T *weight)
      : weight_(weight) {}

  HOSTDEVICE inline T operator()(const T x, const T y) const {
    return x + weight_[0] * (y - x);
  }
};

template <typename T, typename Context>
void LerpKernel(const Context &ctx,
                const DenseTensor &x,
                const DenseTensor &y,
                const DenseTensor &weight,
                DenseTensor *out) {
  PADDLE_ENFORCE_GT(
      x.numel(),
      0,
      common::errors::InvalidArgument("LerpKernel's input x must not empty."));

  PADDLE_ENFORCE_GT(
      y.numel(),
      0,
      common::errors::InvalidArgument("LerpKernel's input y must not empty."));

  int rank = out->dims().size();
  PADDLE_ENFORCE_GE(
      rank,
      0,
      common::errors::InvalidArgument(
          "The number of dimensions for LerpOp must be "
          "greater than or equal to 0, but the value received is %d.",
          rank));

  ctx.template Alloc<T>(out);
  std::vector<DenseTensor *> outputs = {out};

  std::vector<const DenseTensor *> inputs;
  if (weight.numel() == 1) {
    const T *weight_ptr = weight.data<T>();
    inputs.reserve(2);
    inputs.emplace_back(&x);
    inputs.emplace_back(&y);
    auto functor = LerpScalarDirectCUDAFunctor<T>(weight_ptr);
    phi::funcs::BroadcastKernel<T>(ctx, inputs, &outputs, functor);
  } else {
    inputs.reserve(3);
    auto functor = LerpElementWiseDirectCUDAFunctor<T>();
    DenseTensor b_min = phi::EmptyLike<T>(ctx, *out);
    if (x.dims().size() != y.dims().size() &&
        weight.dims().size() != y.dims().size()) {
      if (x.dims().size() < y.dims().size() &&
          x.dims().size() < weight.dims().size()) {
        // x broadcast to b_min
        ExpandKernel<T, Context>(
            ctx, x, common::vectorize(b_min.dims()), &b_min);
        inputs.emplace_back(&b_min);
        inputs.emplace_back(&y);
        inputs.emplace_back(&weight);
      } else if (y.dims().size() < weight.dims().size()) {
        // y broadcast to b_min
        ExpandKernel<T, Context>(
            ctx, y, common::vectorize(b_min.dims()), &b_min);
        inputs.emplace_back(&x);
        inputs.emplace_back(&b_min);
        inputs.emplace_back(&weight);
      } else {
        // weight broadcast to b_min
        ExpandKernel<T, Context>(
            ctx, weight, common::vectorize(b_min.dims()), &b_min);
        inputs.emplace_back(&x);
        inputs.emplace_back(&y);
        inputs.emplace_back(&b_min);
      }
    } else {
      inputs.emplace_back(&x);
      inputs.emplace_back(&y);
      inputs.emplace_back(&weight);
    }
    phi::funcs::BroadcastKernel<T>(ctx, inputs, &outputs, functor);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(lerp,
                   GPU,
                   ALL_LAYOUT,
                   phi::LerpKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   float,
                   double) {}
