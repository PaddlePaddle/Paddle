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

#include "paddle/phi/kernels/cum_grad_kernel.h"
#include "paddle/phi/kernels/cum_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
namespace phi {

template <typename T, typename Context>
void CumsumGradKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& out_grad,
                      const Scalar& axis,
                      bool flatten,
                      bool exclusive,
                      bool reverse,
                      DenseTensor* x_grad) {
  const auto& x_dims = x.dims();
  // If the attribute of flatten is `True`, the cumsum kernel is compose of the
  // operation of flatten and cumsum, need to flatten the tensor of input
  // gradient, and last step need to unflatten the tensor
  if (flatten) {
    x_grad->Resize(out_grad.dims());
  } else {
    x_grad->Resize(x_dims);
  }
  CumsumKernel<T, Context>(
      dev_ctx, out_grad, axis, flatten, exclusive, !reverse, x_grad);
  if (flatten) {
    x_grad->Resize(x_dims);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(cumsum_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::CumsumGradKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t) {}
