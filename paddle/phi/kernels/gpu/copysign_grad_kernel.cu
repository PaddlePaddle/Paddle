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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/gpu/elementwise_grad.h"

namespace phi {

template <typename T>
struct CopySignGradXFunctor {
  inline HOSTDEVICE T operator()(const T x, const T y, const T dout) const {
    return dout * (phi::copysign_func(x, y) / x);
  }
};

template <typename T>
struct CopySignGradYFunctor {
  inline HOSTDEVICE T operator()(const T x, const T y, const T dout) const {
    return static_cast<T>(0);
  }
};

template <typename InT, typename OutT>
struct CopySignGradXYFunctor {
  inline HOSTDEVICE phi::Array<OutT, 2> operator()(const InT x,
                                                   const InT y,
                                                   const InT dout) {
    phi::Array<OutT, 2> outs;
    // dx
    outs[0] = static_cast<OutT>(dout * (phi::copysign_func(x, y)) /
                                static_cast<OutT>(x));
    // dy = 0
    outs[1] = static_cast<OutT>(0);
    return outs;
  }
};

template <typename T, typename Context>
void CopySignGradKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        const DenseTensor& out_grad,
                        DenseTensor* x_grad,
                        DenseTensor* y_grad) {
  const auto place = dev_ctx.GetPlace();
  int axis = -1;
  if (x_grad != nullptr && y_grad != nullptr) {
    std::vector<const DenseTensor*> ins = {&x, &y, &out_grad};
    GetGradXAndYOut<T>(dev_ctx,
                       place,
                       axis,
                       ins,
                       out_grad,
                       x_grad,
                       y_grad,
                       CopySignGradXYFunctor<T, T>());
  } else if (x_grad != nullptr && y_grad == nullptr) {
    std::vector<const DenseTensor*> ins = {&x, &y, &out_grad};
    GetGradXOrYOut<T>(
        dev_ctx, place, axis, ins, out_grad, x_grad, CopySignGradXFunctor<T>());
  } else if (y_grad != nullptr && x_grad == nullptr) {
    std::vector<const DenseTensor*> ins = {&x, &y, &out_grad};
    GetGradXOrYOut<T>(
        dev_ctx, place, axis, ins, out_grad, y_grad, CopySignGradYFunctor<T>());
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(copysign_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::CopySignGradKernel,
                   uint8_t,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
