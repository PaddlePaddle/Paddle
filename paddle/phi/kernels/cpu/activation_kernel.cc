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
DEFINE_CPU_ACTIVATION_KERNEL(Exp, ExpFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Expm1, Expm1Functor)
DEFINE_CPU_ACTIVATION_KERNEL(Reciprocal, ReciprocalFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Square, SquareFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Sqrt, SqrtFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Rsqrt, RsqrtFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Softsign, SoftsignFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Sigmoid, SigmoidFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(LogSigmoid, LogSigmoidFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Log, LogFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Log2, Log2Functor)
DEFINE_CPU_ACTIVATION_KERNEL(Log10, Log10Functor)
DEFINE_CPU_ACTIVATION_KERNEL(Log1p, Log1pFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Round, RoundFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Floor, FloorFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Ceil, CeilFunctor)
DEFINE_CPU_ACTIVATION_KERNEL(Negative, NegativeFunctor)

DEFINE_CPU_ACT_KERNEL_WITH_ONE_ATTRS(LeakyRelu, LeakyReluFunctor, alpha)
DEFINE_CPU_ACT_KERNEL_WITH_ONE_ATTRS(ThresholdedRelu,
                                     ThresholdedReluFunctor,
                                     threshold)
DEFINE_CPU_ACT_KERNEL_WITH_ONE_ATTRS(Relu6, Relu6Functor, threshold)
DEFINE_CPU_ACT_KERNEL_WITH_ONE_ATTRS(Mish, MishFunctor, threshold)
DEFINE_CPU_ACT_KERNEL_WITH_ONE_ATTRS(HardShrink, HardShrinkFunctor, threshold)
DEFINE_CPU_ACT_KERNEL_WITH_ONE_ATTRS(SoftShrink, SoftShrinkFunctor, lambda)
DEFINE_CPU_ACT_KERNEL_WITH_ONE_ATTRS(Elu, ELUFunctor, alpha)
DEFINE_CPU_ACT_KERNEL_WITH_ONE_ATTRS(Swish, SwishFunctor, beta)
DEFINE_CPU_ACT_KERNEL_WITH_ONE_ATTRS(Celu, CELUFunctor, alpha)

DEFINE_CPU_ACT_KERNEL_WITH_TWO_ATTRS(BRelu, BReluFunctor, t_min, t_max)
DEFINE_CPU_ACT_KERNEL_WITH_TWO_ATTRS(STanh, STanhFunctor, scale_a, scale_b)
DEFINE_CPU_ACT_KERNEL_WITH_TWO_ATTRS(Softplus, SoftplusFunctor, beta, threshold)
DEFINE_CPU_ACT_KERNEL_WITH_TWO_ATTRS(HardSigmoid,
                                     HardSigmoidFunctor,
                                     slope,
                                     offset)

template <typename T, typename Context>
void HardSwishKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     float threshold,
                     float scale,
                     float offset,
                     DenseTensor* out) {
  funcs::HardSwishFunctor<T> functor;
  auto attrs = functor.GetAttrs();
  *(attrs[0].second) = threshold;
  *(attrs[1].second) = scale;
  *(attrs[2].second) = offset;
  ActivationImpl<T, Context, funcs::HardSwishFunctor<T>>(
      dev_ctx, x, out, functor);
}

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
PD_REGISTER_ACTIVATION_KERNEL(relu6, Relu6Kernel)
PD_REGISTER_ACTIVATION_KERNEL(hard_shrink, HardShrinkKernel)
PD_REGISTER_ACTIVATION_KERNEL(soft_shrink, SoftShrinkKernel)
PD_REGISTER_ACTIVATION_KERNEL(tanh_shrink, TanhShrinkKernel)
PD_REGISTER_ACTIVATION_KERNEL(elu, EluKernel)
PD_REGISTER_ACTIVATION_KERNEL(silu, SiluKernel)
PD_REGISTER_ACTIVATION_KERNEL(mish, MishKernel)
PD_REGISTER_ACTIVATION_KERNEL(stanh, STanhKernel)
PD_REGISTER_ACTIVATION_KERNEL(reciprocal, ReciprocalKernel)
PD_REGISTER_ACTIVATION_KERNEL(sqrt, SqrtKernel)
PD_REGISTER_ACTIVATION_KERNEL(rsqrt, RsqrtKernel)
PD_REGISTER_ACTIVATION_KERNEL(softplus, SoftplusKernel)

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
PD_REGISTER_ACTIVATION_KERNEL(softsign, SoftsignKernel)
PD_REGISTER_ACTIVATION_KERNEL(sigmoid, SigmoidKernel)
PD_REGISTER_ACTIVATION_KERNEL(logsigmoid, LogSigmoidKernel)
PD_REGISTER_ACTIVATION_KERNEL(hard_sigmoid, HardSigmoidKernel)
PD_REGISTER_ACTIVATION_KERNEL(log, LogKernel)
PD_REGISTER_ACTIVATION_KERNEL(log2, Log2Kernel)
PD_REGISTER_ACTIVATION_KERNEL(log10, Log10Kernel)
PD_REGISTER_ACTIVATION_KERNEL(log1p, Log1pKernel)
PD_REGISTER_ACTIVATION_KERNEL(hard_swish, HardSwishKernel)
PD_REGISTER_ACTIVATION_KERNEL(swish, SwishKernel)
PD_REGISTER_ACTIVATION_KERNEL(round, RoundKernel)
PD_REGISTER_ACTIVATION_KERNEL(floor, FloorKernel)
PD_REGISTER_ACTIVATION_KERNEL(ceil, CeilKernel)
PD_REGISTER_KERNEL(negative,
                   CPU,
                   ALL_LAYOUT,
                   phi::NegativeKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t) {}
PD_REGISTER_ACTIVATION_KERNEL(celu, CeluKernel)
PD_REGISTER_KERNEL(
    pow, CPU, ALL_LAYOUT, phi::PowKernel, float, double, int, int64_t) {}
