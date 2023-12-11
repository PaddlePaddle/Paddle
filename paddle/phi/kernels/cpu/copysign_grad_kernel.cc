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

#include "paddle/phi/kernels/copysign_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/elementwise_grad.h"

namespace phi {

template <typename T>
HOSTDEVICE T compute_copysign_grad_dx(T x, T y, T out, T dout) {
  if (x == static_cast<T>(0))
    return x;
  else
    return static_cast<T>(dout * (phi::copysign_func(x, y) / x));
}

template <typename T>
struct CopySignGradDX {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return compute_copysign_grad_dx<T>(x, y, out, dout);
  }
};

template <typename T>
struct CopySignGradDY {
  HOSTDEVICE T operator()(T x, T y, T out, T dout) const {
    return static_cast<T>(0);
  }
};

template <typename T, typename Context>
void CopySignGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        const DenseTensor& out_grad,
                        DenseTensor* x_grad,
                        DenseTensor* y_grad) {
  funcs::ElementwiseGradPreProcess(out_grad, x_grad);
  int axis = -1;
  phi::funcs::
      ElemwiseGradCompute<Context, T, CopySignGradDX<T>, CopySignGradDY<T>>(
          dev_ctx,
          x,
          y,
          out_grad,
          out_grad,
          axis,
          x_grad,
          y_grad,
          CopySignGradDX<T>(),
          CopySignGradDY<T>());
}
}  // namespace phi

PD_REGISTER_KERNEL(copysign_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::CopySignGradKernel,
                   uint8_t,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
