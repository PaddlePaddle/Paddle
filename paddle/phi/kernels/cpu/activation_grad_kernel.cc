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
    functor_class<T> functor;                                       \
    ActivationGradImpl<T, Context, functor_class<T>>(               \
        dev_ctx, &x, nullptr, &dout, dx, functor);                  \
  }

#define DEFINE_CPU_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DepX( \
    name, functor_class, attr)                          \
  template <typename T, typename Context>               \
  void name##GradKernel(const Context& dev_ctx,         \
                        const DenseTensor& x,           \
                        const DenseTensor& dout,        \
                        float attr,                     \
                        DenseTensor* dx) {              \
    functor_class<T> functor;                           \
    auto attrs = functor.GetAttrs();                    \
    *(attrs[0].second) = attr;                          \
    ActivationGradImpl<T, Context, functor_class<T>>(   \
        dev_ctx, &x, nullptr, &dout, dx, functor);      \
  }

#define DEFINE_CPU_ACT_GRAD_KERNEL_WITH_TWO_ATTRS_DepX( \
    name, functor_class, attr1, attr2)                  \
  template <typename T, typename Context>               \
  void name##GradKernel(const Context& dev_ctx,         \
                        const DenseTensor& x,           \
                        const DenseTensor& dout,        \
                        float attr1,                    \
                        float attr2,                    \
                        DenseTensor* dx) {              \
    functor_class<T> functor;                           \
    auto attrs = functor.GetAttrs();                    \
    *(attrs[0].second) = attr1;                         \
    *(attrs[1].second) = attr2;                         \
    ActivationGradImpl<T, Context, functor_class<T>>(   \
        dev_ctx, &x, nullptr, &dout, dx, functor);      \
  }

#define DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepOut(name, functor_class) \
  template <typename T, typename Context>                             \
  void name##GradKernel(const Context& dev_ctx,                       \
                        const DenseTensor& out,                       \
                        const DenseTensor& dout,                      \
                        DenseTensor* dx) {                            \
    functor_class<T> functor;                                         \
    ActivationGradImpl<T, Context, functor_class<T>>(                 \
        dev_ctx, nullptr, &out, &dout, dx, functor);                  \
  }

#define DEFINE_CPU_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DepOut( \
    name, functor_class, attr)                            \
  template <typename T, typename Context>                 \
  void name##GradKernel(const Context& dev_ctx,           \
                        const DenseTensor& out,           \
                        const DenseTensor& dout,          \
                        float attr,                       \
                        DenseTensor* dx) {                \
    functor_class<T> functor;                             \
    auto attrs = functor.GetAttrs();                      \
    *(attrs[0].second) = attr;                            \
    ActivationGradImpl<T, Context, functor_class<T>>(     \
        dev_ctx, nullptr, &out, &dout, dx, functor);      \
  }

DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepX(Cos, funcs::CosGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepX(Tan, funcs::TanGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepX(Acos, funcs::AcosGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepX(Sin, funcs::SinGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepX(Asin, funcs::AsinGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepX(Atan, funcs::AtanGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepX(Sinh, funcs::SinhGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepX(Cosh, funcs::CoshGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepX(Asinh, funcs::AsinhGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepX(Acosh, funcs::AcoshGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepX(Atanh, funcs::AtanhGradFunctor);

DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepOut(Relu, funcs::ReluGradFunctor);
DEFINE_CPU_ACTIVATION_GRAD_KERNEL_DepOut(Tanh, funcs::TanhGradFunctor);

DEFINE_CPU_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DepX(LeakyRelu,
                                               funcs::LeakyReluGradFunctor,
                                               alpha);
DEFINE_CPU_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DepX(
    ThresholdedRelu, funcs::ThresholdedReluGradFunctor, threshold);

DEFINE_CPU_ACT_GRAD_KERNEL_WITH_TWO_ATTRS_DepX(BRelu,
                                               funcs::BReluGradFunctor,
                                               t_min,
                                               t_max);

}  // namespace phi

PD_REGISTER_KERNEL(
    relu_grad, CPU, ALL_LAYOUT, phi::ReluGradKernel, float, double) {}

#define PD_REGISTER_ACTIVATION_GRAD_KERNEL(name, func) \
  PD_REGISTER_KERNEL(name, CPU, ALL_LAYOUT, phi::func, float, double) {}

#define PD_REGISTER_ACTIVATION_DOUBLE_GRAD_KERNEL(name, func) \
  PD_REGISTER_KERNEL(                                         \
      name, CPU, ALL_LAYOUT, phi::func, float, double, phi::dtype::float16) {}

PD_REGISTER_ACTIVATION_GRAD_KERNEL(sin_grad, SinGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(cos_grad, CosGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(tan_grad, TanGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(acos_grad, AcosGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(asin_grad, AsinGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(atan_grad, AtanGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(sinh_grad, SinhGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(cosh_grad, CoshGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(asinh_grad, AsinhGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(acosh_grad, AcoshGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(atanh_grad, AtanhGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(tanh_grad, TanhGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(brelu_grad, BReluGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(leaky_relu_grad, LeakyReluGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(thresholded_relu_grad,
                                   ThresholdedReluGradKernel)

PD_REGISTER_ACTIVATION_DOUBLE_GRAD_KERNEL(relu_double_grad,
                                          ReluDoubleGradKernel)
PD_REGISTER_ACTIVATION_DOUBLE_GRAD_KERNEL(tanh_double_grad,
                                          TanhDoubleGradKernel)
PD_REGISTER_ACTIVATION_DOUBLE_GRAD_KERNEL(leaky_relu_double_grad,
                                          LeakyReluDoubleGradKernel)

PD_REGISTER_KERNEL(tanh_triple_grad,
                   CPU,
                   ALL_LAYOUT,
                   phi::TanhTripleGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
