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

#include "paddle/phi/kernels/cpu/elementwise.h"
#include "paddle/phi/api/ext/dispatch.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/elementwise_kernel_impl.h"

namespace phi {

#define DEFINE_CPU_ELEMENTWISE_OP(name)                                     \
  template <typename T, typename Context>                                   \
  void name##RawKernel(const Context& dev_ctx,                              \
                       const DenseTensor& x,                                \
                       const DenseTensor& y,                                \
                       int axis,                                            \
                       DenseTensor* out) {                                  \
    dev_ctx.template Alloc<T>(out);                                         \
    if (x.dims() == y.dims()) {                                             \
      SameDimsElementwiseCompute<SameDims##name##Functor<CPUContext, T>>()( \
          dev_ctx, x, y, out);                                              \
    } else {                                                                \
      auto x_dims = x.dims();                                               \
      auto y_dims = y.dims();                                               \
      if (x_dims.size() >= y_dims.size()) {                                 \
        funcs::ElementwiseCompute<funcs::name##Functor<T>, T>(              \
            dev_ctx, x, y, axis, funcs::name##Functor<T>(), out);           \
      } else {                                                              \
        funcs::ElementwiseCompute<funcs::Inverse##name##Functor<T>, T>(     \
            dev_ctx, x, y, axis, funcs::Inverse##name##Functor<T>(), out);  \
      }                                                                     \
    }                                                                       \
  }

template <typename T, typename Context>
void DivideRawKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     int axis,
                     DenseTensor* out) {
  // allocate memory for out
  dev_ctx.template Alloc<T>(out);
  if (x.dims() == y.dims() && std::is_floating_point<T>::value) {
    SameDimsElementwiseCompute<SameDimsDivideFunctor<CPUContext, T>>()(
        dev_ctx, x, y, out);
  } else {
    auto x_dims = x.dims();
    auto y_dims = y.dims();
    if (x_dims.size() >= y_dims.size()) {
      funcs::ElementwiseCompute<funcs::DivideFunctor<T>, T>(
          dev_ctx, x, y, axis, funcs::DivideFunctor<T>(), out);
    } else {
      funcs::ElementwiseCompute<funcs::InverseDivideFunctor<T>, T>(
          dev_ctx, x, y, axis, funcs::InverseDivideFunctor<T>(), out);
    }
  }
}

template <typename T, typename Context>
void MaximumRawKernel(const Context& dev_ctx,
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
void MinimumRawKernel(const Context& dev_ctx,
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
void ModuloRawKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     int axis,
                     DenseTensor* out) {
  // allocate memory for out
  dev_ctx.template Alloc<T>(out);
  auto x_dims = x.dims();
  auto y_dims = y.dims();
  if (x_dims.size() >= y_dims.size()) {
    funcs::ElementwiseCompute<funcs::ModuloFunctor<T>, T>(
        dev_ctx, x, y, axis, funcs::ModuloFunctor<T>(), out);
  } else {
    funcs::ElementwiseCompute<funcs::InverseModuloFunctor<T>, T>(
        dev_ctx, x, y, axis, funcs::InverseModuloFunctor<T>(), out);
  }
}

template <typename T, typename Context>
void FloorDivideRawKernel(const Context& dev_ctx,
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
void ElementwisePowRawKernel(const Context& dev_ctx,
                             const DenseTensor& x,
                             const DenseTensor& y,
                             int axis,
                             DenseTensor* out) {
  // allocate memory for out
  dev_ctx.template Alloc<T>(out);
  funcs::ElementwiseCompute<funcs::ElementwisePowFunctor<T>, T>(
      dev_ctx, x, y, axis, funcs::ElementwisePowFunctor<T>(), out);
}
// Create the definition of Add
DEFINE_CPU_ELEMENTWISE_OP(Add)

// Create the definition of Subtract
DEFINE_CPU_ELEMENTWISE_OP(Subtract)

// Create the definition of Multiply
DEFINE_CPU_ELEMENTWISE_OP(Multiply)

}  // namespace phi

using complex64 = ::phi::dtype::complex<float>;
using complex128 = ::phi::dtype::complex<double>;

// NOTE(chenweihang): using bfloat16 will cause redefine with xpu bfloat16
// using bfloat16 = ::phi::dtype::bfloat16;

PD_REGISTER_KERNEL(
    fmax, CPU, ALL_LAYOUT, phi::FMaxKernel, float, double, int, int64_t) {}

PD_REGISTER_KERNEL(
    fmin, CPU, ALL_LAYOUT, phi::FMinKernel, float, double, int, int64_t) {}

PD_REGISTER_KERNEL(add_raw,
                   CPU,
                   ALL_LAYOUT,
                   phi::AddRawKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   complex64,
                   complex128) {}
PD_REGISTER_KERNEL(subtract_raw,
                   CPU,
                   ALL_LAYOUT,
                   phi::SubtractRawKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t,
                   complex64,
                   complex128,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(divide_raw,
                   CPU,
                   ALL_LAYOUT,
                   phi::DivideRawKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   complex64,
                   complex128) {}
PD_REGISTER_KERNEL(multiply_raw,
                   CPU,
                   ALL_LAYOUT,
                   phi::MultiplyRawKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   bool,
                   complex64,
                   complex128,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(maximum_raw,
                   CPU,
                   ALL_LAYOUT,
                   phi::MaximumRawKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(minimum_raw,
                   CPU,
                   ALL_LAYOUT,
                   phi::MinimumRawKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(modulo_raw,
                   CPU,
                   ALL_LAYOUT,
                   phi::ModuloRawKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
PD_REGISTER_KERNEL(floor_divide_raw,
                   CPU,
                   ALL_LAYOUT,
                   phi::FloorDivideRawKernel,
                   int,
                   int64_t) {}
PD_REGISTER_KERNEL(elementwise_pow_raw,
                   CPU,
                   ALL_LAYOUT,
                   phi::ElementwisePowRawKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
