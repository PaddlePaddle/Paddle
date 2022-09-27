//   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/elementwise_grad_kernel.h"

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/gpu/elementwise_grad.h"
#include "paddle/phi/kernels/impl/elementwise_grad_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void MaximumGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       const DenseTensor& dout,
                       int axis,
                       DenseTensor* dx,
                       DenseTensor* dy) {
  const auto place = dev_ctx.GetPlace();

  if (dx != nullptr && dy != nullptr) {
    std::vector<const DenseTensor*> ins = {&x, &y, &dout};
    GetGradXAndYOut<ElementwiseType::kTernary, T>(
        dev_ctx,
        place,
        axis,
        ins,
        dout,
        dx,
        dy,
        funcs::MaxGradXYFunctor<T, T>());
  } else if (dx != nullptr && dy == nullptr) {
    std::vector<const DenseTensor*> ins = {&x, &y, &dout};
    GetGradXOrYOut<ElementwiseType::kBinary, T>(
        dev_ctx, place, axis, ins, dout, dx, funcs::MaxGradXFunctor<T>());
  } else if (dy != nullptr && dx == nullptr) {
    std::vector<const DenseTensor*> ins = {&x, &y, &dout};
    GetGradXOrYOut<ElementwiseType::kTernary, T>(
        dev_ctx, place, axis, ins, dout, dy, funcs::MaxGradYFunctor<T>());
  }
}

template <typename T, typename Context>
void MinimumGradKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       const DenseTensor& dout,
                       int axis,
                       DenseTensor* dx,
                       DenseTensor* dy) {
  const auto place = dev_ctx.GetPlace();
  if (dx != nullptr && dy != nullptr) {
    std::vector<const DenseTensor*> ins = {&x, &y, &dout};
    GetGradXAndYOut<ElementwiseType::kTernary, T>(
        dev_ctx,
        place,
        axis,
        ins,
        dout,
        dx,
        dy,
        funcs::MinGradXYFunctor<T, T>());
  } else if (dx != nullptr && dy == nullptr) {
    std::vector<const DenseTensor*> ins = {&x, &y, &dout};
    GetGradXOrYOut<ElementwiseType::kBinary, T>(
        dev_ctx, place, axis, ins, dout, dx, funcs::MinGradXFunctor<T>());
  } else if (dy != nullptr && dx == nullptr) {
    std::vector<const DenseTensor*> ins = {&x, &y, &dout};
    GetGradXOrYOut<ElementwiseType::kTernary, T>(
        dev_ctx, place, axis, ins, dout, dy, funcs::MinGradYFunctor<T>());
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(fmax_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ElementwiseFMaxGradKernel,
                   float,
                   double,
                   int,
                   phi::dtype::float16,
                   int64_t) {}

PD_REGISTER_KERNEL(fmin_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ElementwiseFMinGradKernel,
                   float,
                   double,
                   int,
                   phi::dtype::float16,
                   int64_t) {}

PD_REGISTER_KERNEL(maximum_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MaximumGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(minimum_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::MinimumGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(elementwise_heaviside_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ElementwiseHeavisideGradKernel,
                   float,
                   double,
                   int,
                   phi::dtype::float16,
                   int64_t) {}

PD_REGISTER_KERNEL(elementwise_pow_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ElementwisePowGradKernel,
                   float,
                   double,
                   int,
                   phi::dtype::float16,
                   int64_t) {}
