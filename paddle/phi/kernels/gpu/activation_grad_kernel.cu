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

#define DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepX(name, functor_class) \
  template <typename T, typename Context>                           \
  void name##GradKernel(const Context& dev_ctx,                     \
                        const DenseTensor& x,                       \
                        const DenseTensor& dout,                    \
                        DenseTensor* dx) {                          \
    funcs::functor_class<T> functor;                                \
    ActivationGradGPUImpl<T, Context, funcs::functor_class<T>>(     \
        dev_ctx, &x, nullptr, &dout, dx, functor);                  \
  }

#define DEFINE_GPU_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DepX(         \
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

#define DEFINE_GPU_ACT_GRAD_KERNEL_WITH_TWO_ATTRS_DepX(         \
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

#define DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepOut(name, functor_class) \
  template <typename T, typename Context>                             \
  void name##GradKernel(const Context& dev_ctx,                       \
                        const DenseTensor& out,                       \
                        const DenseTensor& dout,                      \
                        DenseTensor* dx) {                            \
    funcs::functor_class<T> functor;                                  \
    ActivationGradGPUImpl<T, Context, funcs::functor_class<T>>(       \
        dev_ctx, nullptr, &out, &dout, dx, functor);                  \
  }

#define DEFINE_GPU_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DepOut(       \
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

DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepOut(Relu, CudaReluGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepOut(Tanh, CudaTanhGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepX(Cos, CudaCosGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepX(Tan, CudaTanGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepX(Acos, CudaAcosGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepX(Sin, CudaSinGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepX(Asin, CudaAsinGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepX(Atan, CudaAtanGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepX(Sinh, CudaSinhGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepX(Cosh, CudaCoshGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepX(Asinh, CudaAsinhGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepX(Acosh, CudaAcoshGradFunctor);
DEFINE_GPU_ACTIVATION_GRAD_KERNEL_DepX(Atanh, CudaAtanhGradFunctor);

DEFINE_GPU_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DepOut(Relu6,
                                                 CudaRelu6GradFunctor,
                                                 threshold)

    DEFINE_GPU_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DepX(LeakyRelu,
                                                   CudaLeakyReluGradFunctor,
                                                   alpha)
        DEFINE_GPU_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DepX(
            ThresholdedRelu, CudaThresholdedReluGradFunctor, threshold)

            DEFINE_GPU_ACT_GRAD_KERNEL_WITH_TWO_ATTRS_DepX(BRelu,
                                                           CudaBReluGradFunctor,
                                                           t_min,
                                                           t_max)

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
PD_REGISTER_ACTIVATION_GRAD_KERNEL(relu6_grad, Relu6GradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(leaky_relu_grad, LeakyReluGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(leaky_relu_double_grad,
                                   LeakyReluDoubleGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(thresholded_relu_grad,
                                   ThresholdedReluGradKernel)
