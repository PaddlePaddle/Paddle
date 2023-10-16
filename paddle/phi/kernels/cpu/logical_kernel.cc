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

#include "paddle/phi/kernels/logical_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/funcs/logical_functor.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/phi/common/transform.h"

namespace phi {

template <typename T, typename Context, typename Functor>
void LogicalKernelImpl(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       DenseTensor* out) {
  Functor binary_func;
  funcs::ElementwiseCompute<Functor, T, bool>(dev_ctx, x, y, binary_func, out);
}

template <typename T, typename Context, typename Functor>
void InplaceLogicalKernelImpl(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& y,
                              DenseTensor* out) {
  Functor binary_func;
  auto x_origin = x;
  out->set_type(phi::DataType::BOOL);
  funcs::ElementwiseCompute<Functor, T, bool>(
      dev_ctx, x_origin, y, binary_func, out);
}

#define DEFINE_LOGICAL_BINARY_KERNEL(type)                                    \
  template <typename T, typename Context>                                     \
  void Logical##type##Kernel(const Context& dev_ctx,                          \
                             const DenseTensor& x,                            \
                             const DenseTensor& y,                            \
                             DenseTensor* out) {                              \
    if (out->IsSharedWith(x)) {                                               \
      InplaceLogicalKernelImpl<T, Context, funcs::Logical##type##Functor<T>>( \
          dev_ctx, x, y, out);                                                \
    } else {                                                                  \
      LogicalKernelImpl<T, Context, funcs::Logical##type##Functor<T>>(        \
          dev_ctx, x, y, out);                                                \
    }                                                                         \
  }

DEFINE_LOGICAL_BINARY_KERNEL(And)
DEFINE_LOGICAL_BINARY_KERNEL(Or)
DEFINE_LOGICAL_BINARY_KERNEL(Xor)
#undef DEFINE_LOGICAL_BINARY_KERNEL

template <typename T, typename Context>
void LogicalNotKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      DenseTensor* out) {
  funcs::LogicalNotFunctor<T> unary_func;

  phi::Transform<Context> trans;
  if (out->IsSharedWith(x)) {
    auto x_origin = x;
    out->set_type(phi::DataType::BOOL);
    auto* out_ptr = dev_ctx.template Alloc<bool>(out);
    trans(dev_ctx,
          x_origin.data<T>(),
          x_origin.data<T>() + x_origin.numel(),
          out_ptr,
          unary_func);
  } else {
    auto* out_ptr = dev_ctx.template Alloc<bool>(out);
    trans(dev_ctx, x.data<T>(), x.data<T>() + x.numel(), out_ptr, unary_func);
  }
}

}  // namespace phi

#define REGISTER_LOGICAL_CPU_KERNEL(logical_and, func_type) \
  PD_REGISTER_KERNEL(logical_and,                           \
                     CPU,                                   \
                     ALL_LAYOUT,                            \
                     phi::Logical##func_type##Kernel,       \
                     float,                                 \
                     double,                                \
                     bool,                                  \
                     int64_t,                               \
                     int,                                   \
                     int8_t,                                \
                     phi::dtype::complex<float>,            \
                     phi::dtype::complex<double>,           \
                     int16_t) {                             \
    kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);   \
  }

REGISTER_LOGICAL_CPU_KERNEL(logical_and, And)
REGISTER_LOGICAL_CPU_KERNEL(logical_or, Or)
REGISTER_LOGICAL_CPU_KERNEL(logical_not, Not)
REGISTER_LOGICAL_CPU_KERNEL(logical_xor, Xor)
