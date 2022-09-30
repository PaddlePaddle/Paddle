// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/kernels/activation_grad_kernel.h"
#include "paddle/phi/kernels/gelu_grad_kernel.h"

#include "paddle/phi/backends/onednn/onednn_context.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"

namespace phi {

#define DEFINE_ONEDNN_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPX( \
    name, functor_class, attr)                             \
  template <typename T, typename Context>                  \
  void name##GradKernel(const Context& dev_ctx,            \
                        const DenseTensor& x,              \
                        const DenseTensor& dout,           \
                        float attr,                        \
                        DenseTensor* dx) {                 \
    functor_class<T> functor;                              \
    functor(dev_ctx, x, dout, attr, 0, dx);                \
  }

#define DEFINE_ONEDNN_ACTIVATION_GRAD_KERNEL_DEPOUT(name, functor_class) \
  template <typename T, typename Context>                                \
  void name##GradKernel(const Context& dev_ctx,                          \
                        const DenseTensor& out,                          \
                        const DenseTensor& dout,                         \
                        DenseTensor* dx) {                               \
    functor_class<T> functor;                                            \
    functor(dev_ctx, out, dout, 0, 0, dx);                               \
  }

template <typename T>
void eltwise_grad(const OneDNNContext& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& dout,
                  float alpha,
                  float beta,
                  DenseTensor* dx,
                  dnnl::algorithm algorithm) {
  funcs::ActivationOneDNNHandler<T> handler(algorithm,
                                            alpha,
                                            beta,
                                            dev_ctx.GetEngine(),
                                            dev_ctx.GetPlace(),
                                            &x,
                                            &dout);

  auto src_memory_p = handler.AcquireBackwardSrcMemory(&x);
  auto diff_dst_memory_p = handler.AcquireDiffDstMemory(&dout);
  auto diff_src_memory_p = handler.AcquireDiffSrcMemory(dx);
  auto activation_backward_p = handler.AcquireBackwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();
  activation_backward_p->execute(astream,
                                 {{DNNL_ARG_SRC, *src_memory_p},
                                  {DNNL_ARG_DIFF_DST, *diff_dst_memory_p},
                                  {DNNL_ARG_DIFF_SRC, *diff_src_memory_p}});
  astream.wait();

  dx->set_mem_desc(diff_src_memory_p->get_desc());
}

template <typename T>
void eltwise_grad_use_out(const OneDNNContext& dev_ctx,
                          const DenseTensor& out,
                          const DenseTensor& dout,
                          float alpha,
                          float beta,
                          DenseTensor* dx,
                          dnnl::algorithm algorithm) {
  funcs::ActivationOneDNNHandler<T> handler(algorithm,
                                            alpha,
                                            beta,
                                            dev_ctx.GetEngine(),
                                            dev_ctx.GetPlace(),
                                            &out,
                                            &dout);

  auto dst_memory_p = handler.AcquireBackwardSrcMemory(&out);
  auto diff_dst_memory_p = handler.AcquireDiffDstMemory(&dout);
  auto diff_src_memory_p = handler.AcquireDiffSrcMemory(dx);
  auto activation_backward_p = handler.AcquireBackwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();
  activation_backward_p->execute(astream,
                                 {{DNNL_ARG_DST, *dst_memory_p},
                                  {DNNL_ARG_DIFF_DST, *diff_dst_memory_p},
                                  {DNNL_ARG_DIFF_SRC, *diff_src_memory_p}});
  astream.wait();

  dx->set_mem_desc(diff_src_memory_p->get_desc());
}

template <typename T, dnnl::algorithm algorithm>
struct OneDNNActivationGradFunc : public funcs::BaseActivationFunctor<T> {
  void operator()(const OneDNNContext& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& dout,
                  float alpha,
                  float beta,
                  DenseTensor* dx) const {
    eltwise_grad<T>(dev_ctx, x, dout, alpha, beta, dx, algorithm);
  }
};

template <typename T, dnnl::algorithm algorithm>
struct OneDNNActivationGradUseOutFunc : public funcs::BaseActivationFunctor<T> {
  void operator()(const OneDNNContext& dev_ctx,
                  const DenseTensor& out,
                  const DenseTensor& dout,
                  float alpha,
                  float beta,
                  DenseTensor* dx) const {
    eltwise_grad_use_out<T>(dev_ctx, out, dout, alpha, beta, dx, algorithm);
  }
};

template <typename T>
using AbsOneDNNGradFunctor =
    OneDNNActivationGradFunc<T, dnnl::algorithm::eltwise_abs>;

template <typename T>
using EluOneDNNGradUseOutFunctor = OneDNNActivationGradUseOutFunc<
    T,
    dnnl::algorithm::eltwise_elu_use_dst_for_bwd>;

template <typename T>
using ExpOneDNNGradUseOutFunctor = OneDNNActivationGradUseOutFunc<
    T,
    dnnl::algorithm::eltwise_exp_use_dst_for_bwd>;

template <typename T>
using HardSwishOneDNNGradFunctor =
    OneDNNActivationGradFunc<T, dnnl::algorithm::eltwise_hardswish>;

template <typename T>
using MishOneDNNGradFunctor =
    OneDNNActivationGradFunc<T, dnnl::algorithm::eltwise_mish>;

template <typename T>
using GeluTanhOneDNNGradFunctor =
    OneDNNActivationGradFunc<T, dnnl::algorithm::eltwise_gelu_tanh>;

template <typename T>
using GeluErfOneDNNGradFunctor =
    OneDNNActivationGradFunc<T, dnnl::algorithm::eltwise_gelu_erf>;

template <typename T>
using ReluOneDNNGradFunctor =
    OneDNNActivationGradFunc<T, dnnl::algorithm::eltwise_relu>;

template <typename T>
using Relu6OneDNNGradUseOutFunctor = OneDNNActivationGradUseOutFunc<
    T,
    dnnl::algorithm::eltwise_clip_v2_use_dst_for_bwd>;

template <typename T>
using SigmoidOneDNNGradUseOutFunctor = OneDNNActivationGradUseOutFunc<
    T,
    dnnl::algorithm::eltwise_logistic_use_dst_for_bwd>;

template <typename T>
using SqrtOneDNNGradUseOutFunctor = OneDNNActivationGradUseOutFunc<
    T,
    dnnl::algorithm::eltwise_sqrt_use_dst_for_bwd>;

template <typename T>
using SwishOneDNNGradFunctor =
    OneDNNActivationGradFunc<T, dnnl::algorithm::eltwise_swish>;

template <typename T>
using TanhOneDNNGradUseOutFunctor = OneDNNActivationGradUseOutFunc<
    T,
    dnnl::algorithm::eltwise_tanh_use_dst_for_bwd>;

DEFINE_ONEDNN_ACTIVATION_GRAD_KERNEL_DEPOUT(Abs, AbsOneDNNGradFunctor);
DEFINE_ONEDNN_ACTIVATION_GRAD_KERNEL_DEPOUT(Exp, ExpOneDNNGradUseOutFunctor);
DEFINE_ONEDNN_ACTIVATION_GRAD_KERNEL_DEPOUT(Relu, ReluOneDNNGradFunctor);
DEFINE_ONEDNN_ACTIVATION_GRAD_KERNEL_DEPOUT(Sigmoid,
                                            SigmoidOneDNNGradUseOutFunctor);
DEFINE_ONEDNN_ACTIVATION_GRAD_KERNEL_DEPOUT(Sqrt, SqrtOneDNNGradUseOutFunctor);
DEFINE_ONEDNN_ACTIVATION_GRAD_KERNEL_DEPOUT(Tanh, TanhOneDNNGradUseOutFunctor);

DEFINE_ONEDNN_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPX(LeakyRelu,
                                                  ReluOneDNNGradFunctor,
                                                  alpha);
DEFINE_ONEDNN_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPX(Mish,
                                                  MishOneDNNGradFunctor,
                                                  threshold);
DEFINE_ONEDNN_ACT_GRAD_KERNEL_WITH_ONE_ATTRS_DEPX(Swish,
                                                  SwishOneDNNGradFunctor,
                                                  beta);

template <typename T, typename Context>
void EluGradKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& out,
                   const DenseTensor& dout,
                   float alpha,
                   DenseTensor* dx) {
  EluOneDNNGradUseOutFunctor<T> functor;
  functor(dev_ctx, out, dout, alpha, 0, dx);
}

template <typename T, typename Context>
void GeluGradKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& out_grad,
                    bool approximate,
                    DenseTensor* x_grad) {
  if (approximate) {
    GeluTanhOneDNNGradFunctor<T> functor;
    functor(dev_ctx, x, out_grad, 0, 0, x_grad);
  } else {
    GeluErfOneDNNGradFunctor<T> functor;
    functor(dev_ctx, x, out_grad, 0, 0, x_grad);
  }
}

template <typename T, typename Context>
void HardSwishGradKernel(const Context& dev_ctx,
                         const DenseTensor& x,
                         const DenseTensor& dout,
                         float threshold,
                         float scale,
                         float offset,
                         DenseTensor* dx) {
  HardSwishOneDNNGradFunctor<T> functor;
  functor(dev_ctx, x, dout, threshold, 0, dx);
}

template <typename T, typename Context>
void Relu6GradKernel(const Context& dev_ctx,
                     const DenseTensor& out,
                     const DenseTensor& dout,
                     float threshold,
                     DenseTensor* dx) {
  Relu6OneDNNGradUseOutFunctor<T> functor;
  functor(dev_ctx, out, dout, 0, threshold, dx);
}

}  // namespace phi

PD_REGISTER_KERNEL(relu_grad,
                   OneDNN,
                   ALL_LAYOUT,
                   phi::ReluGradKernel,
                   float,
                   phi::dtype::bfloat16) {}

#define PD_REGISTER_ACTIVATION_GRAD_KERNEL(name, func) \
  PD_REGISTER_KERNEL(                                  \
      name, OneDNN, ALL_LAYOUT, phi::func, float, phi::dtype::bfloat16) {}

PD_REGISTER_ACTIVATION_GRAD_KERNEL(abs_grad, AbsGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(elu_grad, EluGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(exp_grad, ExpGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(gelu_grad, GeluGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(hard_swish_grad, HardSwishGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(leaky_relu_grad, LeakyReluGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(mish_grad, MishGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(relu6_grad, Relu6GradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(sigmoid_grad, SigmoidGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(sqrt_grad, SqrtGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(swish_grad, SwishGradKernel)
PD_REGISTER_ACTIVATION_GRAD_KERNEL(tanh_grad, TanhGradKernel)
