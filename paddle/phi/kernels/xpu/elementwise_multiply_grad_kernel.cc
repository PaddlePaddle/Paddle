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

#include "paddle/phi/kernels/elementwise_multiply_grad_kernel.h"

#include <memory>
#include <string>

#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/xpu/elementwise.h"

namespace phi {

template <typename T, typename Context>
void MultiplyGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        const DenseTensor& dout,
                        int axis,
                        DenseTensor* dx,
                        DenseTensor* dy) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  funcs::ElementwiseGradPreProcess(dout, dx);
  auto f = [](xpu::Context* ctx,
              const XPUType* x,
              const XPUType* y,
              const XPUType* z,
              const XPUType* dz,
              XPUType* dy,
              XPUType* dx,
              const std::vector<int>& xshape,
              const std::vector<int>& yshape) {
    return xpu::broadcast_mul_grad<XPUType>(
        ctx, x, y, z, dz, dy, dx, xshape, yshape);
  };

  XPUElementwiseGrad<T, XPUType>(dev_ctx, x, y, dout, axis, dx, dy, f, true);
}

}  // namespace phi

PD_REGISTER_KERNEL(multiply_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::MultiplyGradKernel,
                   phi::dtype::float16,
                   phi::dtype::bfloat16,
                   float) {}
