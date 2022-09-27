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

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/bitwise_functors.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"

// See Note [ Why still include the fluid headers? ]
#include "paddle/fluid/platform/transform.h"

namespace phi {

#define DEFINE_BITWISE_KERNEL(op_type)                                    \
  template <typename T, typename Context>                                 \
  void Bitwise##op_type##Kernel(const Context& dev_ctx,                   \
                                const DenseTensor& x,                     \
                                const DenseTensor& y,                     \
                                DenseTensor* out) {                       \
    funcs::Bitwise##op_type##Functor<T> func;                             \
    funcs::ElementwiseCompute<funcs::Bitwise##op_type##Functor<T>, T, T>( \
        dev_ctx, x, y, -1, func, out);                                    \
  }

DEFINE_BITWISE_KERNEL(And)
DEFINE_BITWISE_KERNEL(Or)
DEFINE_BITWISE_KERNEL(Xor)
#undef DEFINE_BITWISE_KERNEL

template <typename T, typename Context>
void BitwiseNotKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      DenseTensor* out) {
  const T* x_data = x.data<T>();
  T* out_data = dev_ctx.template Alloc<T>(out);
  size_t numel = x.numel();
  funcs::BitwiseNotFunctor<T> func;
  paddle::platform::Transform<Context> trans;
  trans(dev_ctx, x_data, x_data + numel, out_data, func);
}

}  // namespace phi

PD_REGISTER_KERNEL(bitwise_and,
                   CPU,
                   ALL_LAYOUT,
                   phi::BitwiseAndKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(bitwise_or,
                   CPU,
                   ALL_LAYOUT,
                   phi::BitwiseOrKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(bitwise_xor,
                   CPU,
                   ALL_LAYOUT,
                   phi::BitwiseXorKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(bitwise_not,
                   CPU,
                   ALL_LAYOUT,
                   phi::BitwiseNotKernel,
                   bool,
                   uint8_t,
                   int8_t,
                   int16_t,
                   int,
                   int64_t) {}
