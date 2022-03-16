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
#include "paddle/phi/kernels/funcs/activation_functor.h"
#include "paddle/phi/kernels/impl/activation_impl.h"

namespace phi {

#define DEFINE_CPU_ACTIVATION_KERNEL(name, functor_class)                \
  template <typename T, typename Context>                                \
  void name##Kernel(                                                     \
      const Context& dev_ctx, const DenseTensor& x, DenseTensor* out) {  \
    functor_class functor;                                               \
    ActivationImpl<T, Context, functor_class>(dev_ctx, x, out, functor); \
  }

#define DEFINE_CPU_ACT_KERNEL_WITH_ONE_ATTRS(name, functor_class, attr)     \
  template <typename T, typename Context>                                   \
  void name##Kernel(const Context& dev_ctx,                                 \
                    const DenseTensor& x,                                   \
                    float attr,                                             \
                    DenseTensor* out) {                                     \
    functor_class<T> functor;                                               \
    auto attrs = functor.GetAttrs();                                        \
    *(attrs[0].second) = attr;                                              \
    ActivationImpl<T, Context, functor_class<T>>(dev_ctx, x, out, functor); \
  }

#define DEFINE_CPU_ACT_KERNEL_WITH_TWO_ATTRS(                               \
    name, functor_class, attr1, attr2)                                      \
  template <typename T, typename Context>                                   \
  void name##Kernel(const Context& dev_ctx,                                 \
                    const DenseTensor& x,                                   \
                    float attr1,                                            \
                    float attr2,                                            \
                    DenseTensor* out) {                                     \
    functor_class<T> functor;                                               \
    auto attrs = functor.GetAttrs();                                        \
    *(attrs[0].second) = attr1;                                             \
    *(attrs[1].second) = attr2;                                             \
    ActivationImpl<T, Context, functor_class<T>>(dev_ctx, x, out, functor); \
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
DEFINE_CPU_ACTIVATION_KERNEL(Tanh, funcs::TanhFunctor<T>)
DEFINE_CPU_ACTIVATION_KERNEL(Exp, funcs::ExpFunctor<T>)
DEFINE_CPU_ACTIVATION_KERNEL(Expm1, funcs::Expm1Functor<T>)
DEFINE_CPU_ACTIVATION_KERNEL(Reciprocal, funcs::ReciprocalFunctor<T>)
DEFINE_CPU_ACTIVATION_KERNEL(Square, funcs::SquareFunctor<T>)
DEFINE_CPU_ACTIVATION_KERNEL(Sqrt, funcs::SqrtFunctor<T>)
DEFINE_CPU_ACTIVATION_KERNEL(Rsqrt, funcs::RsqrtFunctor<T>)

DEFINE_CPU_ACTIVATION_KERNEL(Softsign, funcs::SoftsignFunctor<T>)

DEFINE_CPU_ACT_KERNEL_WITH_ONE_ATTRS(LeakyRelu, funcs::LeakyReluFunctor, alpha)
DEFINE_CPU_ACT_KERNEL_WITH_ONE_ATTRS(ThresholdedRelu,
                                     funcs::ThresholdedReluFunctor,
                                     threshold)
// DEFINE_CPU_ACT_KERNEL_WITH_ONE_ATTRS(Mish, funcs::MishFunctor, threshold)
DEFINE_CPU_ACT_KERNEL_WITH_TWO_ATTRS(BRelu, funcs::BReluFunctor, t_min, t_max)
DEFINE_CPU_ACT_KERNEL_WITH_TWO_ATTRS(STanh,
                                     funcs::STanhFunctor,
                                     scale_a,
                                     scale_b)
// DEFINE_CPU_ACT_KERNEL_WITH_TWO_ATTRS(Softplus,
//                                      funcs::SoftplusFunctor,
//                                      beta,
//                                      threshold)

}  // namespace phi
PD_REGISTER_KERNEL(relu, CPU, ALL_LAYOUT, phi::ReluKernel, float, double) {}

#define PD_REGISTER_ACTIVATION_KERNEL(name, func) \
  PD_REGISTER_KERNEL(name, CPU, ALL_LAYOUT, phi::func##Kernel, float, double) {}

PD_REGISTER_ACTIVATION_KERNEL(sin, Sin)
PD_REGISTER_ACTIVATION_KERNEL(cos, Cos)
PD_REGISTER_ACTIVATION_KERNEL(tan, Tan)
PD_REGISTER_ACTIVATION_KERNEL(acos, Acos)
PD_REGISTER_ACTIVATION_KERNEL(asin, Asin)
PD_REGISTER_ACTIVATION_KERNEL(atan, Atan)
PD_REGISTER_ACTIVATION_KERNEL(sinh, Sinh)
PD_REGISTER_ACTIVATION_KERNEL(cosh, Cosh)
PD_REGISTER_ACTIVATION_KERNEL(asinh, Asinh)
PD_REGISTER_ACTIVATION_KERNEL(acosh, Acosh)
PD_REGISTER_ACTIVATION_KERNEL(atanh, Atanh)
PD_REGISTER_ACTIVATION_KERNEL(tanh, Tanh)
PD_REGISTER_ACTIVATION_KERNEL(brelu, BRelu)
PD_REGISTER_ACTIVATION_KERNEL(leaky_relu, LeakyRelu)
PD_REGISTER_ACTIVATION_KERNEL(thresholded_relu, ThresholdedRelu)
// PD_REGISTER_ACTIVATION_KERNEL(mish, Mish)
PD_REGISTER_ACTIVATION_KERNEL(stanh, STanh)
PD_REGISTER_ACTIVATION_KERNEL(reciprocal, Reciprocal)
PD_REGISTER_ACTIVATION_KERNEL(sqrt, Sqrt)
PD_REGISTER_ACTIVATION_KERNEL(rsqrt, Rsqrt)
// PD_REGISTER_ACTIVATION_KERNEL(softplus, Softplus)
PD_REGISTER_ACTIVATION_KERNEL(softsign, Softsign)

PD_REGISTER_KERNEL(
    exp, CPU, ALL_LAYOUT, phi::ExpKernel, float, double, int, int64_t) {}
PD_REGISTER_KERNEL(expm1,
                   CPU,
                   ALL_LAYOUT,
                   phi::Expm1Kernel,
                   float,
                   double,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(logit, CPU, ALL_LAYOUT, phi::LogitKernel, float, double) {}
PD_REGISTER_KERNEL(
    square, CPU, ALL_LAYOUT, phi::SquareKernel, float, double, int, int64_t) {}
