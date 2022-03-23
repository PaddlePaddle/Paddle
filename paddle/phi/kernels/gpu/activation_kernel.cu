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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/impl/activation_impl.h"

#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"

namespace phi {

template <typename T, typename Context, typename Functor>
void ActivationGPUImpl(const Context& dev_ctx,
                       const DenseTensor& x,
                       DenseTensor* out,
                       const Functor& functor) {
  PADDLE_ENFORCE_NOT_NULL(out,
                          errors::NotFound("Output Out should not be nullptr"));
  dev_ctx.template Alloc<T>(out);
  std::vector<const DenseTensor*> ins = {&x};
  std::vector<DenseTensor*> outs = {out};
  funcs::ElementwiseKernel<T>(dev_ctx, ins, &outs, functor);
}

#define DEFINE_GPU_ACTIVATION_KERNEL(name, functor_class)               \
  template <typename T, typename Context>                               \
  void name##Kernel(                                                    \
      const Context& dev_ctx, const DenseTensor& x, DenseTensor* out) { \
    funcs::functor_class<T> functor;                                    \
    ActivationGPUImpl<T, Context, funcs::functor_class<T>>(             \
        dev_ctx, x, out, functor);                                      \
  }

#define DEFINE_GPU_ACT_KERNEL_WITH_ONE_ATTRS(name, functor_class, attr) \
  template <typename T, typename Context>                               \
  void name##Kernel(const Context& dev_ctx,                             \
                    const DenseTensor& x,                               \
                    float attr,                                         \
                    DenseTensor* out) {                                 \
    funcs::functor_class<T> functor;                                    \
    auto attrs = functor.GetAttrs();                                    \
    *(attrs[0].second) = attr;                                          \
    ActivationGPUImpl<T, Context, funcs::functor_class<T>>(             \
        dev_ctx, x, out, functor);                                      \
  }

#define DEFINE_GPU_ACT_KERNEL_WITH_TWO_ATTRS(               \
    name, functor_class, attr1, attr2)                      \
  template <typename T, typename Context>                   \
  void name##Kernel(const Context& dev_ctx,                 \
                    const DenseTensor& x,                   \
                    float attr1,                            \
                    float attr2,                            \
                    DenseTensor* out) {                     \
    funcs::functor_class<T> functor;                        \
    auto attrs = functor.GetAttrs();                        \
    *(attrs[0].second) = attr1;                             \
    *(attrs[1].second) = attr2;                             \
    ActivationGPUImpl<T, Context, funcs::functor_class<T>>( \
        dev_ctx, x, out, functor);                          \
  }

DEFINE_GPU_ACTIVATION_KERNEL(Cos, CudaCosFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Tan, CudaTanFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Acos, CudaAcosFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Sin, CudaSinFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Asin, CudaAsinFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Atan, CudaAtanFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Sinh, CudaSinhFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Cosh, CudaCoshFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Asinh, CudaAsinhFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Acosh, CudaAcoshFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Atanh, CudaAtanhFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Relu, CudaReluFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Tanh, CudaTanhFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(TanhShrink, CudaTanhShrinkFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Silu, CudaSiluFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Sigmoid, CudaSigmoidFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(LogSigmoid, CudaLogSigmoidFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Log, CudaLogFunctor)
DEFINE_GPU_ACTIVATION_KERNEL(Log2, CudaLog2Functor)
DEFINE_GPU_ACTIVATION_KERNEL(Log10, CudaLog10Functor)
DEFINE_GPU_ACTIVATION_KERNEL(Log1p, CudaLog1pFunctor)

DEFINE_GPU_ACT_KERNEL_WITH_ONE_ATTRS(LeakyRelu, CudaLeakyReluFunctor, alpha)
DEFINE_GPU_ACT_KERNEL_WITH_ONE_ATTRS(ThresholdedRelu,
                                     CudaThresholdedReluFunctor,
                                     threshold)
DEFINE_GPU_ACT_KERNEL_WITH_ONE_ATTRS(HardShrink,
                                     CudaHardShrinkFunctor,
                                     threshold)
DEFINE_GPU_ACT_KERNEL_WITH_ONE_ATTRS(SoftShrink, CudaSoftShrinkFunctor, lambda)
DEFINE_GPU_ACT_KERNEL_WITH_ONE_ATTRS(Elu, CudaELUFunctor, alpha)

DEFINE_GPU_ACT_KERNEL_WITH_TWO_ATTRS(BRelu, CudaBReluFunctor, t_min, t_max)
DEFINE_GPU_ACT_KERNEL_WITH_TWO_ATTRS(HardSigmoid,
                                     CudaHardSigmoidFunctor,
                                     slope,
                                     offset)

}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(relu,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReluKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(relu,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReluKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif

#define PD_REGISTER_ACTIVATION_KERNEL(name, func) \
  PD_REGISTER_KERNEL(name,                        \
                     GPU,                         \
                     ALL_LAYOUT,                  \
                     phi::func,                   \
                     float,                       \
                     double,                      \
                     phi::dtype::float16,         \
                     phi::dtype::bfloat16) {}

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
PD_REGISTER_ACTIVATION_KERNEL(thresholded_relu, ThresholdedReluKernel)
PD_REGISTER_ACTIVATION_KERNEL(leaky_relu, LeakyReluKernel)
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
