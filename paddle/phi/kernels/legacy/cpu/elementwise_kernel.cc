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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/cpu/elementwise.h"
#include "paddle/phi/kernels/impl/elementwise_kernel_impl.h"

namespace phi {

template <typename T, typename Context>
void MaximumWithAxisKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& y,
                           int axis,
                           DenseTensor* out) {
  // allocate memory for out
  dev_ctx.template Alloc<T>(out);
  funcs::ElementwiseCompute<funcs::MaximumFunctor<T>, T>(
      dev_ctx, x, y, axis, funcs::MaximumFunctor<T>(), out);
}

template <typename T, typename Context>
void MinimumWithAxisKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& y,
                           int axis,
                           DenseTensor* out) {
  // allocate memory for out
  dev_ctx.template Alloc<T>(out);
  funcs::ElementwiseCompute<funcs::MinimumFunctor<T>, T>(
      dev_ctx, x, y, axis, funcs::MinimumFunctor<T>(), out);
}

template <typename T, typename Context>
void RemainderWithAxisKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& y,
                             int axis,
                             DenseTensor* out) {
  // allocate memory for out
  dev_ctx.template Alloc<T>(out);
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  if (x_dims.size() >= y_dims.size()) {
    funcs::ElementwiseCompute<funcs::RemainderFunctor<T>, T>(
        dev_ctx, x, y, axis, funcs::RemainderFunctor<T>(), out);
  } else {
    funcs::ElementwiseCompute<funcs::InverseRemainderFunctor<T>, T>(
        dev_ctx, x, y, axis, funcs::InverseRemainderFunctor<T>(), out);
  }
}

template <typename T, typename Context>
void FloorDivideWithAxisKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& y,
                               int axis,
                               DenseTensor* out) {
  // allocate memory for out
  dev_ctx.template Alloc<T>(out);
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  if (x_dims.size() >= y_dims.size()) {
    funcs::ElementwiseCompute<funcs::FloorDivideFunctor<T>, T>(
        dev_ctx, x, y, axis, funcs::FloorDivideFunctor<T>(), out);
  } else {
    funcs::ElementwiseCompute<funcs::InverseFloorDivideFunctor<T>, T>(
        dev_ctx, x, y, axis, funcs::InverseFloorDivideFunctor<T>(), out);
  }
}

template <typename T, typename Context>
void ElementwisePowWithAxisKernel(const Context& dev_ctx,
                                  const DenseTensor& x,
                                  const DenseTensor& y,
                                  int axis,
                                  DenseTensor* out) {
  // allocate memory for out
  dev_ctx.template Alloc<T>(out);
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  if (x_dims.size() >= y_dims.size()) {
    funcs::ElementwiseCompute<funcs::ElementwisePowFunctor<T>, T>(
        dev_ctx, x, y, axis, funcs::ElementwisePowFunctor<T>(), out);
  } else {
    funcs::ElementwiseCompute<funcs::ElementwiseInversePowFunctor<T>, T>(
        dev_ctx, x, y, axis, funcs::ElementwiseInversePowFunctor<T>(), out);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(maximum_with_axis,
                   CPU,
                   ALL_LAYOUT,
                   phi::MaximumWithAxisKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(minimum_with_axis,
                   CPU,
                   ALL_LAYOUT,
                   phi::MinimumWithAxisKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(remainder_with_axis,
                   CPU,
                   ALL_LAYOUT,
                   phi::RemainderWithAxisKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
PD_REGISTER_KERNEL(floor_divide_with_axis,
                   CPU,
                   ALL_LAYOUT,
                   phi::FloorDivideWithAxisKernel,
                   int,
                   int64_t) {}
PD_REGISTER_KERNEL(elementwise_pow_with_axis,
                   CPU,
                   ALL_LAYOUT,
                   phi::ElementwisePowWithAxisKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
