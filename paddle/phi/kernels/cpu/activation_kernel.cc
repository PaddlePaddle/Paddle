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

#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/activation_impl.h"

namespace phi {

#define DEFINE_CPU_ACTIVATION_KERNEL(name, functor_class)                \
  template <typename T, typename Context>                                \
  void name##Kernel(                                                     \
      const Context& dev_ctx, const DenseTensor& x, DenseTensor* out) {  \
    functor_class functor;                                               \
    ActivationImpl<T, Context, functor_class>(dev_ctx, x, out, functor); \
  }

DEFINE_CPU_ACTIVATION_KERNEL(Sin, funcs::SinFunctor<T>)
DEFINE_CPU_ACTIVATION_KERNEL(Cos, funcs::CosFunctor<T>)
DEFINE_CPU_ACTIVATION_KERNEL(Tan, funcs::TanFunctor<T>)
DEFINE_CPU_ACTIVATION_KERNEL(Asin, funcs::AsinFunctor<T>)
DEFINE_CPU_ACTIVATION_KERNEL(Atan, funcs::AtanFunctor<T>)
DEFINE_CPU_ACTIVATION_KERNEL(Acos, funcs::AcosFunctor<T>)
DEFINE_CPU_ACTIVATION_KERNEL(Sinh, funcs::SinhFunctor<T>)
DEFINE_CPU_ACTIVATION_KERNEL(Cosh, funcs::CoshFunctor<T>)
DEFINE_CPU_ACTIVATION_KERNEL(Asinh, funcs::AsinhFunctor<T>)
DEFINE_CPU_ACTIVATION_KERNEL(Acosh, funcs::AcoshFunctor<T>)
DEFINE_CPU_ACTIVATION_KERNEL(Atanh, funcs::AtanhFunctor<T>)
DEFINE_CPU_ACTIVATION_KERNEL(Relu, funcs::ReluCPUFunctor<T>)

}  // namespace phi
PD_REGISTER_KERNEL(sin, CPU, ALL_LAYOUT, phi::SinKernel, float, double) {}
PD_REGISTER_KERNEL(cos, CPU, ALL_LAYOUT, phi::CosKernel, float, double) {}
PD_REGISTER_KERNEL(tan, CPU, ALL_LAYOUT, phi::TanKernel, float, double) {}
PD_REGISTER_KERNEL(acos, CPU, ALL_LAYOUT, phi::AcosKernel, float, double) {}
PD_REGISTER_KERNEL(asin, CPU, ALL_LAYOUT, phi::AsinKernel, float, double) {}
PD_REGISTER_KERNEL(atan, CPU, ALL_LAYOUT, phi::AtanKernel, float, double) {}
PD_REGISTER_KERNEL(sinh, CPU, ALL_LAYOUT, phi::SinhKernel, float, double) {}
PD_REGISTER_KERNEL(cosh, CPU, ALL_LAYOUT, phi::CoshKernel, float, double) {}
PD_REGISTER_KERNEL(asinh, CPU, ALL_LAYOUT, phi::AsinhKernel, float, double) {}
PD_REGISTER_KERNEL(acosh, CPU, ALL_LAYOUT, phi::AcoshKernel, float, double) {}
PD_REGISTER_KERNEL(atanh, CPU, ALL_LAYOUT, phi::AtanhKernel, float, double) {}
PD_REGISTER_KERNEL(relu, CPU, ALL_LAYOUT, phi::ReluKernel, float, double) {}
