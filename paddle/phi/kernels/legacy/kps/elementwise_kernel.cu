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

#include "paddle/phi/backends/gpu/gpu_context.h"
#ifndef PADDLE_WITH_XPU_KP
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#endif
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/elementwise_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void MaximumRawKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      int axis,
                      DenseTensor* out) {
  std::vector<const DenseTensor*> inputs;
  inputs.reserve(2);
  std::vector<DenseTensor*> outputs;
  outputs.reserve(1);
  inputs.emplace_back(&x);
  inputs.emplace_back(&y);
  outputs.emplace_back(out);
  dev_ctx.template Alloc<T>(out);
  funcs::BroadcastKernel<ElementwiseType::kBinary, T, T>(
      dev_ctx, inputs, &outputs, axis, funcs::MaximumFunctor<T>());
}

template <typename T, typename Context>
void MinimumRawKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const DenseTensor& y,
                      int axis,
                      DenseTensor* out) {
  std::vector<const DenseTensor*> inputs;
  inputs.reserve(2);
  std::vector<DenseTensor*> outputs;
  outputs.reserve(1);
  inputs.emplace_back(&x);
  inputs.emplace_back(&y);
  outputs.emplace_back(out);
  dev_ctx.template Alloc<T>(out);
  funcs::BroadcastKernel<ElementwiseType::kBinary, T, T>(
      dev_ctx, inputs, &outputs, axis, funcs::MinimumFunctor<T>());
}

template <typename T, typename Context>
void RemainderRawKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        int axis,
                        DenseTensor* out) {
  std::vector<const DenseTensor*> inputs;
  inputs.reserve(2);
  std::vector<DenseTensor*> outputs;
  outputs.reserve(1);
  inputs.emplace_back(&x);
  inputs.emplace_back(&y);
  outputs.emplace_back(out);
  dev_ctx.template Alloc<T>(out);
  funcs::BroadcastKernel<ElementwiseType::kBinary, T, T>(
      dev_ctx, inputs, &outputs, axis, funcs::RemainderFunctor<T>());
}

template <typename T, typename Context>
void FloorDivideRawKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& y,
                          int axis,
                          DenseTensor* out) {
  std::vector<const DenseTensor*> inputs;
  inputs.reserve(2);
  std::vector<DenseTensor*> outputs;
  outputs.reserve(1);
  inputs.emplace_back(&x);
  inputs.emplace_back(&y);
  outputs.emplace_back(out);
  dev_ctx.template Alloc<T>(out);
  funcs::BroadcastKernel<ElementwiseType::kBinary, T, T>(
      dev_ctx, inputs, &outputs, axis, funcs::FloorDivideFunctor<T>());
}

template <typename T, typename Context>
void ElementwisePowRawKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& y,
                             int axis,
                             DenseTensor* out) {
  std::vector<const DenseTensor*> inputs;
  inputs.reserve(2);
  std::vector<DenseTensor*> outputs;
  outputs.reserve(1);
  inputs.emplace_back(&x);
  inputs.emplace_back(&y);
  outputs.emplace_back(out);
  dev_ctx.template Alloc<T>(out);
  funcs::BroadcastKernel<ElementwiseType::kBinary, T, T>(
      dev_ctx, inputs, &outputs, axis, funcs::ElementwisePowFunctor<T>());
}

}  // namespace phi

#ifdef PADDLE_WITH_XPU_KP
PD_REGISTER_KERNEL(maximum_raw, KPS, ALL_LAYOUT, phi::MaximumRawKernel, float) {
}
PD_REGISTER_KERNEL(minimum_raw, KPS, ALL_LAYOUT, phi::MinimumRawKernel, float) {
}
PD_REGISTER_KERNEL(
    floor_divide_raw, KPS, ALL_LAYOUT, phi::FloorDivideRawKernel, int) {}
PD_REGISTER_KERNEL(
    elementwise_pow_raw, KPS, ALL_LAYOUT, phi::ElementwisePowRawKernel, float) {
}

#else
using float16 = phi::dtype::float16;
using bfloat16 = phi::dtype::bfloat16;

PD_REGISTER_KERNEL(maximum_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::MaximumRawKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   float16,
                   bfloat16) {}
PD_REGISTER_KERNEL(minimum_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::MinimumRawKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   float16,
                   bfloat16) {}
PD_REGISTER_KERNEL(remainder_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::RemainderRawKernel,
                   float,
                   double,
                   int,
                   float16,
                   int64_t) {}
PD_REGISTER_KERNEL(floor_divide_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::FloorDivideRawKernel,
                   int,
                   int64_t) {}
PD_REGISTER_KERNEL(elementwise_pow_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::ElementwisePowRawKernel,
                   float,
                   double,
                   int,
                   float16,
                   int64_t) {}
#endif
