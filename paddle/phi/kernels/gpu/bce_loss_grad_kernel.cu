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

#include "paddle/phi/kernels/bce_loss_grad_kernel.h"

#include <algorithm>
#include <vector>

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/hostdevice.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"

namespace phi {

template <typename T>
struct BCELossGradFunctor {
  T one;
  T eps;

  HOSTDEVICE inline BCELossGradFunctor() {
    one = static_cast<T>(1.0f);
    eps = static_cast<T>(1e-12);
  }

  HOSTDEVICE inline T operator()(const T x, const T label, const T dout) const {
    T term1 = max((one - x) * x, eps);
    return (dout * (x - label) / term1);
  }
};

template <typename T, typename Context>
void BCELossGradKernel(const Context& dev_ctx,
                       const DenseTensor& input,
                       const DenseTensor& label,
                       const DenseTensor& out_grad,
                       DenseTensor* input_grad) {
  dev_ctx.template Alloc<T>(input_grad);
  std::vector<const DenseTensor*> ins = {&input, &label, &out_grad};
  std::vector<DenseTensor*> outs = {input_grad};
  auto functor = BCELossGradFunctor<T>();
  phi::funcs::ElementwiseKernel<T>(dev_ctx, ins, &outs, functor);
}

}  // namespace phi

PD_REGISTER_KERNEL(
    bce_loss_grad, GPU, ALL_LAYOUT, phi::BCELossGradKernel, float, double) {}
