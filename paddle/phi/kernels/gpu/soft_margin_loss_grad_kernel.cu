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
#include "paddle/phi/kernels/soft_margin_loss_grad_kernel.h"

namespace phi {

template <typename T>
struct SoftMarginLossGradFunctor {
  T one;
  T eps;

  HOSTDEVICE inline SoftMarginLossGradFunctor() {
    one = static_cast<T>(1.0f);
    eps = static_cast<T>(1e-12);
  }

  HOSTDEVICE inline T operator()(const T x, const T label, const T dout) const {
    T term1 = (one + std::exp(-label * x));
    return (dout * (-label * std::exp(-label * x)) / term1);
  }
};

template <typename T, typename Context>
void SoftMarginLossGradKernel(const Context& dev_ctx,
                              const DenseTensor& input,
                              const DenseTensor& label,
                              const DenseTensor& out_grad,
                              DenseTensor* input_grad) {
  dev_ctx.template Alloc<T>(input_grad);
  std::vector<const DenseTensor*> ins = {&input, &label, &out_grad};
  std::vector<DenseTensor*> outs = {input_grad};
  auto functor = SoftMarginLossGradFunctor<T>();
  phi::funcs::ElementwiseKernel<T>(dev_ctx, ins, &outs, functor);
}

}  // namespace phi

PD_REGISTER_KERNEL(soft_margin_loss_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::SoftMarginLossGradKernel,
                   float,
                   double) {}
