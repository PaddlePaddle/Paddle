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

#include "paddle/common/hostdevice.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"

namespace phi {

template <typename T>
struct BCELossGradFunctor {
  using MT = typename phi::dtype::MPTypeTrait<T>::Type;
  MT one = static_cast<MT>(1.0f);
  MT eps = static_cast<MT>(1e-12);

  HOSTDEVICE inline T operator()(const T x, const T label, const T dout) const {
    MT x_mt = static_cast<MT>(x);
    MT term1 = max((one - x_mt) * x_mt, eps);
    return static_cast<T>(static_cast<MT>(dout) *
                          (x_mt - static_cast<MT>(label)) / term1);
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

PD_REGISTER_KERNEL(bce_loss_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::BCELossGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
