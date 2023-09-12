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
// limitation

#include "paddle/phi/kernels/logical_kernel.h"
#ifdef PADDLE_WITH_XPU_KP
#include "paddle/phi/backends/xpu/xpu_context.h"
#else
#include "paddle/phi/backends/gpu/gpu_context.h"
#endif

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/logical_functor.h"

namespace phi {

template <typename T, typename Context, typename Functor>
void LogicalKernelImpl(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       DenseTensor* out) {
  dev_ctx.template Alloc<bool>(out);
  Functor binary_func;
  std::vector<const DenseTensor*> ins = {&x, &y};
  std::vector<DenseTensor*> outs = {out};
  funcs::BroadcastKernel<bool>(dev_ctx, ins, &outs, binary_func);
}

template <typename T, typename Context, typename Functor>
void InplaceLogicalKernelImpl(const Context& dev_ctx,
                              const DenseTensor& x,
                              const DenseTensor& y,
                              DenseTensor* out) {
  auto x_origin = x;
  dev_ctx.template Alloc<bool>(out);
  out->set_type(phi::DataType::BOOL);
  Functor binary_func;
  std::vector<const DenseTensor*> ins = {&x_origin, &y};
  std::vector<DenseTensor*> outs = {out};
  funcs::BroadcastKernel<bool>(dev_ctx, ins, &outs, binary_func);
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
  if (!out->IsSharedWith(x)) {
    dev_ctx.template Alloc<bool>(out);
    funcs::LogicalNotFunctor<T> unary_func;
    std::vector<const DenseTensor*> ins = {&x};
    std::vector<DenseTensor*> outs = {out};
    funcs::BroadcastKernel<bool>(dev_ctx, ins, &outs, unary_func);
  } else {
    auto x_origin = x;
    out->set_type(phi::DataType::BOOL);
    dev_ctx.template Alloc<bool>(out);
    funcs::LogicalNotFunctor<T> unary_func;
    std::vector<const DenseTensor*> ins = {&x_origin};
    std::vector<DenseTensor*> outs = {out};
    funcs::BroadcastKernel<bool>(dev_ctx, ins, &outs, unary_func);
  }
}

}  // namespace phi

#ifdef PADDLE_WITH_XPU_KP
PD_REGISTER_KERNEL(logical_and, KPS, ALL_LAYOUT, phi::LogicalAndKernel, int) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
PD_REGISTER_KERNEL(logical_or, KPS, ALL_LAYOUT, phi::LogicalOrKernel, int) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
PD_REGISTER_KERNEL(logical_not, KPS, ALL_LAYOUT, phi::LogicalNotKernel, int) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
PD_REGISTER_KERNEL(logical_xor, KPS, ALL_LAYOUT, phi::LogicalXorKernel, int) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
#else

#define REGISTER_LOGICAL_CUDA_KERNEL(logical_and, func_type) \
  PD_REGISTER_KERNEL(logical_and,                            \
                     KPS,                                    \
                     ALL_LAYOUT,                             \
                     phi::Logical##func_type##Kernel,        \
                     float,                                  \
                     phi::dtype::float16,                    \
                     phi::dtype::bfloat16,                   \
                     double,                                 \
                     bool,                                   \
                     int64_t,                                \
                     int,                                    \
                     int8_t,                                 \
                     phi::dtype::complex<float>,             \
                     phi::dtype::complex<double>,            \
                     int16_t) {                              \
    kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);    \
  }

REGISTER_LOGICAL_CUDA_KERNEL(logical_and, And)
REGISTER_LOGICAL_CUDA_KERNEL(logical_or, Or)
REGISTER_LOGICAL_CUDA_KERNEL(logical_not, Not)
REGISTER_LOGICAL_CUDA_KERNEL(logical_xor, Xor)
#endif
