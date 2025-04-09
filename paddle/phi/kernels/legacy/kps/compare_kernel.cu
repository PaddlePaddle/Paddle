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

#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/impl/compare_kernel_impl.h"

#ifdef PADDLE_WITH_XPU_KP
#include "paddle/phi/backends/xpu/xpu_context.h"
#else
#include <thrust/fill.h>

#include <vector>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/compare_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/gpu/reduce.h"
#include "paddle/phi/kernels/primitive/functor_primitives.h"
#endif

namespace phi {

template <typename T, typename Context, typename Functor>
inline void CompareRawKernelImpl(const Context& ctx,
                                 const DenseTensor& x,
                                 const DenseTensor& y,
                                 int axis,
                                 DenseTensor* out) {
  ctx.template Alloc<bool>(out);
  out->set_type(phi::DataType::BOOL);
  if (out->numel() == 0) return;
  std::vector<const DenseTensor*> ins{&x, &y};
  std::vector<DenseTensor*> outs{out};
  funcs::BroadcastKernel<bool>(ctx, ins, &outs, Functor(), axis);
}

template <typename T, typename Context>
void LessThanRawKernel(const Context& ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       int axis,
                       DenseTensor* out) {
  CompareRawKernelImpl<T, Context, funcs::LessThanFunctor<T>>(
      ctx, x, y, axis, out);
}

template <typename T, typename Context>
void LessEqualRawKernel(const Context& ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        int axis,
                        DenseTensor* out) {
  CompareRawKernelImpl<T, Context, funcs::LessEqualFunctor<T>>(
      ctx, x, y, axis, out);
}

template <typename T, typename Context>
void GreaterThanRawKernel(const Context& ctx,
                          const DenseTensor& x,
                          const DenseTensor& y,
                          int axis,
                          DenseTensor* out) {
  CompareRawKernelImpl<T, Context, funcs::GreaterThanFunctor<T>>(
      ctx, x, y, axis, out);
}

template <typename T, typename Context>
void GreaterEqualRawKernel(const Context& ctx,
                           const DenseTensor& x,
                           const DenseTensor& y,
                           int axis,
                           DenseTensor* out) {
  CompareRawKernelImpl<T, Context, funcs::GreaterEqualFunctor<T>>(
      ctx, x, y, axis, out);
}

template <typename T, typename Context>
void EqualRawKernel(const Context& ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    int axis,
                    DenseTensor* out) {
  CompareRawKernelImpl<T, Context, funcs::EqualFunctor<T>>(
      ctx, x, y, axis, out);
}

template <typename T, typename Context>
void NotEqualRawKernel(const Context& ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       int axis,
                       DenseTensor* out) {
  CompareRawKernelImpl<T, Context, funcs::NotEqualFunctor<T>>(
      ctx, x, y, axis, out);
}

}  // namespace phi

#ifdef PADDLE_WITH_XPU_KP
PD_REGISTER_KERNEL(
    less_than_raw, KPS, ALL_LAYOUT, phi::LessThanRawKernel, int) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
PD_REGISTER_KERNEL(
    less_equal_raw, KPS, ALL_LAYOUT, phi::LessEqualRawKernel, int) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
PD_REGISTER_KERNEL(
    greater_than_raw, KPS, ALL_LAYOUT, phi::GreaterThanRawKernel, int) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
PD_REGISTER_KERNEL(
    greater_equal_raw, KPS, ALL_LAYOUT, phi::GreaterEqualRawKernel, int) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
PD_REGISTER_KERNEL(equal_raw, KPS, ALL_LAYOUT, phi::EqualRawKernel, int) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
PD_REGISTER_KERNEL(
    not_equal_raw, KPS, ALL_LAYOUT, phi::NotEqualRawKernel, int) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

#else

PD_REGISTER_KERNEL(less_than_raw,
                   KPS,
                   ALL_LAYOUT,
                   phi::LessThanRawKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}

#define PD_REGISTER_COMPARE_RAW_KERNEL(name, func)        \
  PD_REGISTER_KERNEL(name##_raw,                          \
                     KPS,                                 \
                     ALL_LAYOUT,                          \
                     phi::func##RawKernel,                \
                     bool,                                \
                     uint8_t,                             \
                     int16_t,                             \
                     int,                                 \
                     int8_t,                              \
                     int64_t,                             \
                     float,                               \
                     double,                              \
                     phi::dtype::float16,                 \
                     phi::dtype::bfloat16) {              \
    kernel->OutputAt(0).SetDataType(phi::DataType::BOOL); \
  }

#define PD_REGISTER_COMPLEX_COMPARE_RAW_KERNEL(name, func) \
  PD_REGISTER_KERNEL(name##_raw,                           \
                     KPS,                                  \
                     ALL_LAYOUT,                           \
                     phi::func##RawKernel,                 \
                     bool,                                 \
                     uint8_t,                              \
                     int16_t,                              \
                     int,                                  \
                     int8_t,                               \
                     int64_t,                              \
                     phi::dtype::complex<float>,           \
                     phi::dtype::complex<double>,          \
                     float,                                \
                     double,                               \
                     phi::dtype::float16,                  \
                     phi::dtype::bfloat16) {               \
    kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);  \
  }

PD_REGISTER_COMPARE_RAW_KERNEL(less_equal, LessEqual)
PD_REGISTER_COMPARE_RAW_KERNEL(greater_than, GreaterThan)
PD_REGISTER_COMPARE_RAW_KERNEL(greater_equal, GreaterEqual)

PD_REGISTER_COMPLEX_COMPARE_RAW_KERNEL(equal, Equal)
PD_REGISTER_COMPLEX_COMPARE_RAW_KERNEL(not_equal, NotEqual)

#endif
