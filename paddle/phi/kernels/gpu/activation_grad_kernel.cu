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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/impl/activation_grad_impl.h"

#include "paddle/fluid/platform/device/gpu/gpu_device_function.h"

namespace phi {

template <typename T, typename Context, typename Functor>
void ActivationGradGPUImpl(const Context& dev_ctx,
                           const DenseTensor* x,
                           const DenseTensor* out,
                           const DenseTensor* d_out,
                           DenseTensor* d_x,
                           const Functor& functor) {
  if (static_cast<int>(Functor::FwdDeps()) &
      static_cast<int>(funcs::ActBwdOpFwdDeps::kDepOut)) {
    PADDLE_ENFORCE_NOT_NULL(
        out, errors::NotFound("The input DenseTensor Out can not be nullptr"));
  }
  PADDLE_ENFORCE_NOT_NULL(
      d_out, errors::NotFound("The input DenseTensor dOut can not be nullptr"));
  PADDLE_ENFORCE_NOT_NULL(
      d_x, errors::NotFound("The output DenseTensor dX can not be nullptr"));
  if (!out) {
    out = d_out;  // fake out
  }
  if (static_cast<int>(Functor::FwdDeps()) &
      static_cast<int>(funcs::ActBwdOpFwdDeps::kDepX)) {
    PADDLE_ENFORCE_NOT_NULL(
        x, errors::NotFound("The input DenseTensor X can not be nullptr"));
  } else {
    VLOG(10) << "Inplace activation of Op Functor: " << typeid(Functor).name();
    x = d_x;
  }

  dev_ctx.template Alloc<T>(d_x);

  std::vector<const DenseTensor*> ins = {d_out};
  std::vector<DenseTensor*> outs = {d_x};

  if (static_cast<int>(Functor::FwdDeps()) ==
      static_cast<int>(funcs::ActBwdOpFwdDeps::kDepOut)) {
    // Only need forward output Out
    ins.push_back(out);
    funcs::ElementwiseKernel<T>(dev_ctx, ins, &outs, functor);
  } else if (static_cast<int>(Functor::FwdDeps()) ==
             static_cast<int>(funcs::ActBwdOpFwdDeps::kDepX)) {
    // Only need forward input X
    ins.push_back(x);
    funcs::ElementwiseKernel<T>(dev_ctx, ins, &outs, functor);
  } else {
    funcs::ElementwiseKernel<T>(dev_ctx, ins, &outs, functor);
  }
}

#define DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DEPX(name, functor_class) \
  template <typename T, typename Context>                           \
  void name##GradKernel(const Context& dev_ctx,                     \
                        const DenseTensor& x,                       \
                        const DenseTensor& dout,                    \
                        DenseTensor* dx) {                          \
    funcs::functor_class<T> functor;                                \
    ActivationGradGPUImpl<T, Context, funcs::functor_class<T>>(     \
        dev_ctx, &x, nullptr, &dout, dx, functor);                  \
  }

#define DEFINE_GPU_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPX(         \
    name, functor_class, attr)                                  \
  template <typename T, typename Context>                       \
  void name##GradKernel(const Context& dev_ctx,                 \
                        const DenseTensor& x,                   \
                        const DenseTensor& dout,                \
                        float attr,                             \
                        DenseTensor* dx) {                      \
    funcs::functor_class<T> functor;                            \
    auto attrs = functor.GetAttrs();                            \
    *(attrs[0].second) = attr;                                  \
    ActivationGradGPUImpl<T, Context, funcs::functor_class<T>>( \
        dev_ctx, &x, nullptr, &dout, dx, functor);              \
  }

#define DEFINE_GPU_ACT_GRAD_KERNEL_WITH_TWO_ATTRS_DEPX(         \
    name, functor_class, attr1, attr2)                          \
  template <typename T, typename Context>                       \
  void name##GradKernel(const Context& dev_ctx,                 \
                        const DenseTensor& x,                   \
                        const DenseTensor& dout,                \
                        float attr1,                            \
                        float attr2,                            \
                        DenseTensor* dx) {                      \
    funcs::functor_class<T> functor;                            \
    auto attrs = functor.GetAttrs();                            \
    *(attrs[0].second) = attr1;                                 \
    *(attrs[1].second) = attr2;                                 \
    ActivationGradGPUImpl<T, Context, funcs::functor_class<T>>( \
        dev_ctx, &x, nullptr, &dout, dx, functor);              \
  }

#define DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DEPOUT(name, functor_class) \
  template <typename T, typename Context>                             \
  void name##GradKernel(const Context& dev_ctx,                       \
                        const DenseTensor& out,                       \
                        const DenseTensor& dout,                      \
                        DenseTensor* dx) {                            \
    funcs::functor_class<T> functor;                                  \
    ActivationGradGPUImpl<T, Context, funcs::functor_class<T>>(       \
        dev_ctx, nullptr, &out, &dout, dx, functor);                  \
  }

#define DEFINE_GPU_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPOUT(       \
    name, functor_class, attr)                                  \
  template <typename T, typename Context>                       \
  void name##GradKernel(const Context& dev_ctx,                 \
                        const DenseTensor& out,                 \
                        const DenseTensor& dout,                \
                        float attr,                             \
                        DenseTensor* dx) {                      \
    funcs::functor_class<T> functor;                            \
    auto attrs = functor.GetAttrs();                            \
    *(attrs[0].second) = attr;                                  \
    ActivationGradGPUImpl<T, Context, funcs::functor_class<T>>( \
        dev_ctx, nullptr, &out, &dout, dx, functor);            \
  }

#define DEFINE_GPU_ACT_GRAD_KERNEL_WITH_TWO_ATTRS_DEPOUT(       \
    name, functor_class, attr1, attr2)                          \
  template <typename T, typename Context>                       \
  void name##GradKernel(const Context& dev_ctx,                 \
                        const DenseTensor& out,                 \
                        const DenseTensor& dout,                \
                        float attr1,                            \
                        float attr2,                            \
                        DenseTensor* dx) {                      \
    funcs::functor_class<T> functor;                            \
    auto attrs = functor.GetAttrs();                            \
    *(attrs[0].second) = attr1;                                 \
    *(attrs[1].second) = attr2;                                 \
    ActivationGradGPUImpl<T, Context, funcs::functor_class<T>>( \
        dev_ctx, nullptr, &out, &dout, dx, functor);            \
  }

DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DEPOUT(Relu, CudaReluGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DEPOUT(Tanh, CudaTanhGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DEPOUT(Sigmoid, CudaSigmoidGradFunctor);

DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DEPX(Cos, CudaCosGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DEPX(Tan, CudaTanGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DEPX(Acos, CudaAcosGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DEPX(Sin, CudaSinGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DEPX(Asin, CudaAsinGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DEPX(Atan, CudaAtanGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DEPX(Sinh, CudaSinhGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DEPX(Cosh, CudaCoshGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DEPX(Asinh, CudaAsinhGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DEPX(Acosh, CudaAcoshGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DEPX(Atanh, CudaAtanhGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DEPX(TanhShrink, CudaTanhShrinkGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DEPX(Silu, CudaSiluGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DEPX(LogSigmoid, CudaLogSigmoidGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DEPX(Log, CudaLogGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DEPX(Log2, CudaLog2GradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DEPX(Log10, CudaLog10GradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DEPX(Log1p, CudaLog1pGradFunctor);

DEFINE_GPU_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPX(LeakyRelu,
                                               CudaLeakyReluGradFunctor,
                                               alpha);
DEFINE_GPU_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPX(ThresholdedRelu,
                                               CudaThresholdedReluGradFunctor,
                                               threshold);
DEFINE_GPU_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPX(SoftShrink,
                                               CudaSoftShrinkGradFunctor,
                                               lambda);
DEFINE_GPU_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPX(HardShrink,
                                               CudaHardShrinkGradFunctor,
                                               threshold);

DEFINE_GPU_ACT_GRAD_KERNEL_WITH_TWO_ATTRS_DEPX(BRelu,
                                               CudaBReluGradFunctor,
                                               t_min,
                                               t_max);

DEFINE_GPU_ACT_GRAD_KERNEL_WITH_TWO_ATTRS_DEPOUT(HardSigmoid,
                                                 CudaHardSigmoidGradFunctor,
                                                 slope,
                                                 offset);

template <typename T, typename Context>
void EluGradKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& out,
                   const DenseTensor& dout,
                   float alpha,
                   DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  std::vector<const DenseTensor*> ins = {&dout, &out};
  std::vector<DenseTensor*> outs = {dx};
  if (alpha > 0) {
    funcs::CudaELUGradFunctor<T> functor;
    functor.alpha = alpha;
    funcs::ElementwiseKernel<T>(dev_ctx, ins, &outs, functor);
  } else {
    funcs::CudaELUGradNegativeAlphaFunctor<T> functor;
    functor.alpha = alpha;
    ins.push_back(&x);
    funcs::ElementwiseKernel<T>(dev_ctx, ins, &outs, functor);
  }
}

}  // namespace phi

#ifdef PADDLE_WITH_HIP
PD_REGISTER_KERNEL(relu_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReluGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
PD_REGISTER_KERNEL(relu_double_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReluDoubleGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
#else
PD_REGISTER_KERNEL(relu_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReluGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
PD_REGISTER_KERNEL(relu_double_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::ReluDoubleGradKernel,
                   float,
                   double,
                   phi::dtype::float16,
                   phi::dtype::bfloat16) {}
#endif

#define PD_REGISTER_ACTIVATION_GRAD_KERNEL(name, func) \
  PD_REGISTER_KERNEL(name,                             \
                     GPU,                              \
                     ALL_LAYOUT,                       \
                     phi::func,                        \
                     float,                            \
                     double,                           \
                     phi::dtype::float16,              \
                     phi::dtype::bfloat16) {}

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
PD_REGISTER_ACTIVATION_GRAD_KERNEL(tanh_double_grad, TanhDoubleGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(tanh_triple_grad, TanhTripleGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(brelu_grad, BReluGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(leaky_relu_grad, LeakyReluGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(leaky_relu_double_grad,
                                   LeakyReluDoubleGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(thresholded_relu_grad,
                                   ThresholdedReluGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(soft_shrink_grad, SoftShrinkGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(hard_shrink_grad, HardShrinkGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(tanh_shrink_grad, TanhShrinkGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(silu_grad, SiluGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(elu_grad, EluGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(elu_double_grad, EluDoubleGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(sigmoid_grad, SigmoidGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(sigmoid_double_grad, SigmoidDoubleGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(sigmoid_triple_grad, SigmoidTripleGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(hard_sigmoid_grad, HardSigmoidGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(logsigmoid_grad, LogSigmoidGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(log_grad, LogGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(log2_grad, Log2GradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(log10_grad, Log10GradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(log1p_grad, Log1pGradKernel)
PD_REGISTER_KERNEL(log_double_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::LogDoubleGradKernel,
                   float,
                   double,
                   phi::dtype::float16) {}
