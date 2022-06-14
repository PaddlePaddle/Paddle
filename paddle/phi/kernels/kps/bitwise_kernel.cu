/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/bitwise_kernel.h"

#ifdef PADDLE_WITH_XPU_KP
#include "paddle/phi/backends/xpu/xpu_context.h"
#else
#include "paddle/phi/backends/gpu/gpu_context.h"
#endif

#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/bitwise_functors.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
namespace phi {

#define DEFINE_BITWISE_KERNEL(op_type)                      \
  template <typename T, typename Context>                   \
  void Bitwise##op_type##Kernel(const Context& dev_ctx,     \
                                const DenseTensor& x,       \
                                const DenseTensor& y,       \
                                DenseTensor* out) {         \
    dev_ctx.template Alloc<T>(out);                         \
    funcs::Bitwise##op_type##Functor<T> func;               \
    std::vector<const DenseTensor*> ins = {&x, &y};         \
    std::vector<DenseTensor*> outs = {out};                 \
    funcs::BroadcastKernel<ElementwiseType::kBinary, T, T>( \
        dev_ctx, ins, &outs, -1, func);                     \
  }

DEFINE_BITWISE_KERNEL(And)
DEFINE_BITWISE_KERNEL(Or)
DEFINE_BITWISE_KERNEL(Xor)
#undef DEFINE_BITWISE_KERNEL

template <typename T, typename Context>
void BitwiseNotKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  std::vector<const DenseTensor*> ins = {&x};
  std::vector<DenseTensor*> outs = {out};
  funcs::BitwiseNotFunctor<T> func;
  funcs::BroadcastKernel<ElementwiseType::kUnary, T, T>(
      dev_ctx, ins, &outs, -1, func);
}

}  // namespace phi

#ifdef PADDLE_WITH_XPU_KP
PD_REGISTER_KERNEL(
    bitwise_and, KPS, ALL_LAYOUT, phi::BitwiseAndKernel, int, bool) {}
PD_REGISTER_KERNEL(
    bitwise_or, KPS, ALL_LAYOUT, phi::BitwiseOrKernel, int, bool) {}
PD_REGISTER_KERNEL(
    bitwise_xor, KPS, ALL_LAYOUT, phi::BitwiseXorKernel, int, bool) {}
PD_REGISTER_KERNEL(
    bitwise_not, KPS, ALL_LAYOUT, phi::BitwiseNotKernel, int, bool) {}

#else
PD_REGISTER_KERNEL(bitwise_and,
                   KPS,
                   ALL_LAYOUT,
                   phi::BitwiseAndKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(bitwise_or,
                   KPS,
                   ALL_LAYOUT,
                   phi::BitwiseOrKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(bitwise_xor,
                   KPS,
                   ALL_LAYOUT,
                   phi::BitwiseXorKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(bitwise_not,
                   KPS,
                   ALL_LAYOUT,
                   phi::BitwiseNotKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

#endif
