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

#include "paddle/phi/kernels/bce_loss_kernel.h"

#include <algorithm>
#include <vector>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/primitive/functor_primitives.h"

namespace phi {

template <typename T>
struct BCELossFunctor {
  T one;
  T neg_100;

  HOSTDEVICE inline BCELossFunctor() {
    one = static_cast<T>(1.0f);
    neg_100 = static_cast<T>(-100.);
  }

  HOSTDEVICE inline T operator()(const T x, const T label) const {
    PADDLE_ENFORCE(
        (x >= static_cast<T>(0)) && (x <= one),
        "Input is expected to be within the interval [0, 1], but recieved %f.",
        x);
    T term1 = max(phi::kps::details::Log(x), neg_100);
    T term2 = max(phi::kps::details::Log(one - x), neg_100);
    return (((label - one) * term2) - (label * term1));
  }
};

template <typename T, typename Context>
void BCELossKernel(const Context& dev_ctx,
                   const DenseTensor& input,
                   const DenseTensor& label,
                   DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  std::vector<const DenseTensor*> ins = {&input, &label};
  std::vector<DenseTensor*> outs = {out};
  auto functor = BCELossFunctor<T>();
  phi::funcs::ElementwiseKernel<T>(dev_ctx, ins, &outs, functor);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    bce_loss, GPU, ALL_LAYOUT, phi::BCELossKernel, float, double) {}
