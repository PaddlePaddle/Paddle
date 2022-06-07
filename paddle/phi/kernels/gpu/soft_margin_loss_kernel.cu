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

#include <algorithm>
#include <vector>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/primitive/functor_primitives.h"
#include "paddle/phi/kernels/soft_margin_loss_kernel.h"

namespace phi {

template <typename T>
struct SoftMarginLossFunctor {
  T one;

  HOSTDEVICE inline SoftMarginLossFunctor() { one = static_cast<T>(1.0f); }

  HOSTDEVICE inline T operator()(const T x, const T label) const {
    T term1 = std::log(one + std::exp(-label * x));
    return term1;
  }
};

template <typename T, typename Context>
void SoftMarginLossKernel(const Context& dev_ctx,
                          const DenseTensor& input,
                          const DenseTensor& label,
                          DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  std::vector<const DenseTensor*> ins = {&input, &label};
  std::vector<DenseTensor*> outs = {out};
  auto functor = SoftMarginLossFunctor<T>();
  phi::funcs::ElementwiseKernel<T>(dev_ctx, ins, &outs, functor);
}

}  // namespace phi

PD_REGISTER_KERNEL(soft_margin_loss,
                   GPU,
                   ALL_LAYOUT,
                   phi::SoftMarginLossKernel,
                   float,
                   double) {}
