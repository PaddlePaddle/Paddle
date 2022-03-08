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

#include "paddle/phi/kernels/activation_grad_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/activation_grad_impl.h"

namespace phi {

#define DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepX(name, functor_class) \
  template <typename T, typename Context>                           \
  void name##GradKernel(const Context& dev_ctx,                     \
                        const DenseTensor& x,                       \
                        const DenseTensor& dout,                    \
                        DenseTensor* dx) {                          \
    functor_class functor;                                          \
    ActivationGradImpl<T, Context, functor_class>(                  \
        dev_ctx, &x, nullptr, &dout, dx, functor);                  \
  }

#define DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepOut(name, functor_class) \
  template <typename T, typename Context>                             \
  void name##GradKernel(const Context& dev_ctx,                       \
                        const DenseTensor& out,                       \
                        const DenseTensor& dout,                      \
                        DenseTensor* dx) {                            \
    functor_class functor;                                            \
    ActivationGradImpl<T, Context, functor_class>(                    \
        dev_ctx, nullptr, &out, &dout, dx, functor);                  \
  }

DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepX(Cos, funcs::CosGradFunctor<T>);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepX(Tan, funcs::TanGradFunctor<T>);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepX(Acos, funcs::AcosGradFunctor<T>);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepX(Sin, funcs::SinGradFunctor<T>);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepX(Asin, funcs::AsinGradFunctor<T>);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepX(Atan, funcs::AtanGradFunctor<T>);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepX(Sinh, funcs::SinhGradFunctor<T>);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepX(Cosh, funcs::CoshGradFunctor<T>);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepX(Asinh, funcs::AsinhGradFunctor<T>);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepX(Acosh, funcs::AcoshGradFunctor<T>);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepX(Atanh, funcs::AtanhGradFunctor<T>);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepOut(Relu, funcs::ReluGradFunctor<T>);

}  // namespace phi

PD_REGISTER_KERNEL(
    cos_grad, CPU, ALL_LAYOUT, phi::CosGradKernel, float, double) {}
PD_REGISTER_KERNEL(
    tan_grad, CPU, ALL_LAYOUT, phi::TanGradKernel, float, double) {}
PD_REGISTER_KERNEL(
    acos_grad, CPU, ALL_LAYOUT, phi::AcosGradKernel, float, double) {}
PD_REGISTER_KERNEL(
    sin_grad, CPU, ALL_LAYOUT, phi::SinGradKernel, float, double) {}
PD_REGISTER_KERNEL(
    asin_grad, CPU, ALL_LAYOUT, phi::AsinGradKernel, float, double) {}
PD_REGISTER_KERNEL(
    atan_grad, CPU, ALL_LAYOUT, phi::AtanGradKernel, float, double) {}
PD_REGISTER_KERNEL(
    sinh_grad, CPU, ALL_LAYOUT, phi::SinhGradKernel, float, double) {}
PD_REGISTER_KERNEL(
    cosh_grad, CPU, ALL_LAYOUT, phi::CoshGradKernel, float, double) {}
PD_REGISTER_KERNEL(
    asinh_grad, CPU, ALL_LAYOUT, phi::AsinhGradKernel, float, double) {}
PD_REGISTER_KERNEL(
    acosh_grad, CPU, ALL_LAYOUT, phi::AcoshGradKernel, float, double) {}
PD_REGISTER_KERNEL(
    atanh_grad, CPU, ALL_LAYOUT, phi::AtanhGradKernel, float, double) {}
PD_REGISTER_KERNEL(
    relu_grad, CPU, ALL_LAYOUT, phi::ReluGradKernel, float, double) {}
PD_REGISTER_KERNEL(relu_double_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::ReluDoubleGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
