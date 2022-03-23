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

#define DEFINE_CPU_ACTIVATION_KERNEL(name, functor_class)               \
  template <typename T, typename Context>                               \
  void name##Kernel(                                                    \
      const Context& dev_ctx, const DenseTensor& x, DenseTensor* out) { \
    funcs::functor_class<T> functor;                                    \
    ActivationImpl<T, Context, funcs::functor_class<T>>(                \
        dev_ctx, x, out, functor);                                      \
  }

#define DEFINE_CPU_ACT_KERNEL_WITH_ONE_ATTRS(name, functor_class, attr) \
  template <typename T, typename Context>                               \
  void name##Kernel(const Context& dev_ctx,                             \
                    const DenseTensor& x,                               \
                    float attr,                                         \
                    DenseTensor* out) {                                 \
    funcs::functor_class<T> functor;                                    \
    auto attrs = functor.GetAttrs();                                    \
    *(attrs[0].second) = attr;                                          \
    ActivationImpl<T, Context, funcs::functor_class<T>>(                \
        dev_ctx, x, out, functor);                                      \
  }

#define DEFINE_CPU_ACT_KERNEL_WITH_TWO_ATTRS(            \
    name, functor_class, attr1, attr2)                   \
  template <typename T, typename Context>                \
  void name##Kernel(const Context& dev_ctx,              \
                    const DenseTensor& x,                \
                    float attr1,                         \
                    float attr2,                         \
                    DenseTensor* out) {                  \
    funcs::functor_class<T> functor;                     \
    auto attrs = functor.GetAttrs();                     \
    *(attrs[0].second) = attr1;                          \
    *(attrs[1].second) = attr2;                          \
    ActivationImpl<T, Context, funcs::functor_class<T>>( \
        dev_ctx, x, out, functor);                       \
  }

DEFINE_CPU_ACTIVATION_KERNEL(Sin, SinFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Cos, CosFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Tan, TanFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Asin, AsinFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Atan, AtanFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Acos, AcosFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Sinh, SinhFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Cosh, CoshFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Asinh, AsinhFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Acosh, AcoshFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Atanh, AtanhFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Relu, ReluCPUFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Tanh, TanhFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(TanhShrink, TanhShrinkFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Silu, SiluFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Sigmoid, SigmoidFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(LogSigmoid, LogSigmoidFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Log, LogFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Log2, Log2Functor)
DEFINE_CPU_ACTIVATION_KERNEL(Log10, Log10Functor)
DEFINE_CPU_ACTIVATION_KERNEL(Log1p, Log1pFunctor)

DEFINE_CPU_ACT_KERNEL_WITH_ONE_ATTRS(LeakyRelu, LeakyReluFunctor, alpha)
DEFINE_CPU_ACT_KERNEL_WITH_ONE_ATTRS(ThresholdedRelu,
                                     ThresholdedReluFunctor,
                                     threshold)
DEFINE_CPU_ACT_KERNEL_WITH_ONE_ATTRS(HardShrink, HardShrinkFunctor, threshold)
DEFINE_CPU_ACT_KERNEL_WITH_ONE_ATTRS(SoftShrink, SoftShrinkFunctor, lambda)
DEFINE_CPU_ACT_KERNEL_WITH_ONE_ATTRS(Elu, ELUFunctor, alpha)

DEFINE_CPU_ACT_KERNEL_WITH_TWO_ATTRS(BRelu, BReluFunctor, t_min, t_max)
DEFINE_CPU_ACT_KERNEL_WITH_TWO_ATTRS(HardSigmoid,
                                     HardSigmoidFunctor,
                                     slope,
                                     offset)

}  // namespace phi
PD_REGISTER_KERNEL(relu, CPU, ALL_LAYOUT, phi::ReluKernel, float, double) {}

#define PD_REGISTER_ACTIVATION_KERNEL(name, func) \
  PD_REGISTER_KERNEL(name, CPU, ALL_LAYOUT, phi::func, float, double) {}

PD_REGISTER_ACTIVATION_KERNEL(sin, SinKernel)
PD_REGISTER_ACTIVATION_KERNEL(cos, CosKernel)
PD_REGISTER_ACTIVATION_KERNEL(tan, TanKernel)
PD_REGISTER_ACTIVATION_KERNEL(acos, AcosKernel)
PD_REGISTER_ACTIVATION_KERNEL(asin, AsinKernel)
PD_REGISTER_ACTIVATION_KERNEL(atan, AtanKernel)
PD_REGISTER_ACTIVATION_KERNEL(sinh, SinhKernel)
PD_REGISTER_ACTIVATION_KERNEL(cosh, CoshKernel)
PD_REGISTER_ACTIVATION_KERNEL(asinh, AsinhKernel)
PD_REGISTER_ACTIVATION_KERNEL(acosh, AcoshKernel)
PD_REGISTER_ACTIVATION_KERNEL(atanh, AtanhKernel)
PD_REGISTER_ACTIVATION_KERNEL(tanh, TanhKernel)
PD_REGISTER_ACTIVATION_KERNEL(brelu, BReluKernel)
PD_REGISTER_ACTIVATION_KERNEL(leaky_relu, LeakyReluKernel)
PD_REGISTER_ACTIVATION_KERNEL(thresholded_relu, ThresholdedReluKernel)
PD_REGISTER_ACTIVATION_KERNEL(hard_shrink, HardShrinkKernel)
PD_REGISTER_ACTIVATION_KERNEL(soft_shrink, SoftShrinkKernel)
PD_REGISTER_ACTIVATION_KERNEL(tanh_shrink, TanhShrinkKernel)
PD_REGISTER_ACTIVATION_KERNEL(elu, EluKernel)
PD_REGISTER_ACTIVATION_KERNEL(silu, SiluKernel)
PD_REGISTER_ACTIVATION_KERNEL(sigmoid, SigmoidKernel)
PD_REGISTER_ACTIVATION_KERNEL(logsigmoid, LogSigmoidKernel)
PD_REGISTER_ACTIVATION_KERNEL(hard_sigmoid, HardSigmoidKernel)
PD_REGISTER_ACTIVATION_KERNEL(log, LogKernel)
PD_REGISTER_ACTIVATION_KERNEL(log2, Log2Kernel)
PD_REGISTER_ACTIVATION_KERNEL(log10, Log10Kernel)
PD_REGISTER_ACTIVATION_KERNEL(log1p, Log1pKernel)
