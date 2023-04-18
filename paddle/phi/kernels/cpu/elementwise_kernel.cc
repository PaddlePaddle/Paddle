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
void MaximumKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  // allocate memory for out
  dev_ctx.template Alloc<T>(out);
  int axis = -1;
  funcs::ElementwiseCompute<funcs::MaximumFunctor<T>, T>(
      dev_ctx, x, y, axis, funcs::MaximumFunctor<T>(), out);
}

template <typename T, typename Context>
void MinimumKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  // allocate memory for out
  dev_ctx.template Alloc<T>(out);
  int axis = -1;
  funcs::ElementwiseCompute<funcs::MinimumFunctor<T>, T>(
      dev_ctx, x, y, axis, funcs::MinimumFunctor<T>(), out);
}

template <typename T, typename Context>
void RemainderKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     DenseTensor* out) {
  // allocate memory for out
  dev_ctx.template Alloc<T>(out);
  int axis = -1;
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
void FloorDivideKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       DenseTensor* out) {
  // allocate memory for out
  dev_ctx.template Alloc<T>(out);
  int axis = -1;
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
void ElementwisePowKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& y,
                          DenseTensor* out) {
  // allocate memory for out
  dev_ctx.template Alloc<T>(out);
  int axis = -1;
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

template <typename T, typename Context>
void HeavisideKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     DenseTensor* out) {
  // allocate memory for out
  dev_ctx.template Alloc<T>(out);
  funcs::ElementwiseCompute<funcs::ElementwiseHeavisideFunctor<T>, T>(
      dev_ctx, x, y, -1, funcs::ElementwiseHeavisideFunctor<T>(), out);
}

}  // namespace phi

using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

// NOTE(chenweihang): using bfloat16 will cause redefine with xpu bfloat16
// using bfloat16 = ::phi::dtype::bfloat16;

PD_REGISTER_KERNEL(
    fmax, CPU, ALL_LAYOUT, phi::FMaxKernel, float, double, int, int64_t) {}

PD_REGISTER_KERNEL(
    fmin, CPU, ALL_LAYOUT, phi::FMinKernel, float, double, int, int64_t) {}

PD_REGISTER_KERNEL(maximum,
                   CPU,
                   ALL_LAYOUT,
                   phi::MaximumKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(minimum,
                   CPU,
                   ALL_LAYOUT,
                   phi::MinimumKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(remainder,
                   CPU,
                   ALL_LAYOUT,
                   phi::RemainderKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
PD_REGISTER_KERNEL(
    floor_divide, CPU, ALL_LAYOUT, phi::FloorDivideKernel, int, int64_t) {}
PD_REGISTER_KERNEL(elementwise_pow,
                   CPU,
                   ALL_LAYOUT,
                   phi::ElementwisePowKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}

PD_REGISTER_KERNEL(heaviside,
                   CPU,
                   ALL_LAYOUT,
                   phi::HeavisideKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
