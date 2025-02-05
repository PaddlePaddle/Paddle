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

#include "paddle/phi/kernels/activation_kernel.h"
#include "paddle/phi/kernels/gelu_grad_kernel.h"

#include "paddle/phi/backends/onednn/onednn_context.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"
namespace phi {

#define DEFINE_ONEDNN_ACTIVATION_KERNEL(name, functor_class)            \
  template <typename T, typename Context>                               \
  void name##Kernel(                                                    \
      const Context& dev_ctx, const DenseTensor& x, DenseTensor* out) { \
    functor_class<T> functor;                                           \
    functor(dev_ctx, x, 0, 0, out);                                     \
  }

#define DEFINE_ONEDNN_ACT_KERNEL_WITH_ONE_ATTRS(name, functor_class, attr) \
  template <typename T, typename Context>                                  \
  void name##Kernel(const Context& dev_ctx,                                \
                    const DenseTensor& x,                                  \
                    float attr,                                            \
                    DenseTensor* out) {                                    \
    functor_class<T> functor;                                              \
    functor(dev_ctx, x, attr, 0, out);                                     \
  }

template <typename T>
void EltwiseForward(const OneDNNContext& dev_ctx,
                    const DenseTensor& x,
                    float alpha,
                    float beta,
                    DenseTensor* out,
                    dnnl::algorithm algorithm) {
  PADDLE_ENFORCE_EQ(dev_ctx.GetPlace().GetType() == phi::AllocationType::CPU,
                    true,
                    common::errors::PreconditionNotMet(
                        "Operator DNNL eletwise_forward must use ONEDNNPlace"));

  bool is_inplaced = x.IsSharedBufferWith(*out);

  funcs::ActivationOneDNNHandler<T> handler(
      algorithm, alpha, beta, dev_ctx.GetEngine(), dev_ctx.GetPlace(), &x);

  auto src_memory_p = handler.AcquireSrcMemory(&x);
  std::shared_ptr<dnnl::memory> dst_memory_p = nullptr;
  if (is_inplaced) {
    dst_memory_p = src_memory_p;
    dev_ctx.template Alloc<T>(out);
  } else {
    dst_memory_p = handler.AcquireDstMemory(out);
  }
  auto activation_p = handler.AcquireForwardPrimitive();

  auto& astream = OneDNNContext::tls().get_stream();
  activation_p->execute(
      astream, {{DNNL_ARG_FROM, *src_memory_p}, {DNNL_ARG_TO, *dst_memory_p}});
  astream.wait();

  out->set_mem_desc(dst_memory_p->get_desc());
}

template <typename T, dnnl::algorithm algorithm>
struct OneDNNActivationFunc : public funcs::BaseActivationFunctor<T> {
  void operator()(const OneDNNContext& dev_ctx,
                  const DenseTensor& x,
                  float alpha,
                  float beta,
                  DenseTensor* out) const {
    EltwiseForward<T>(dev_ctx, x, alpha, beta, out, algorithm);
  }
};

template <typename T>
using AbsOneDNNFunctor = OneDNNActivationFunc<T, dnnl::algorithm::eltwise_abs>;

template <typename T>
using EluOneDNNFunctor = OneDNNActivationFunc<T, dnnl::algorithm::eltwise_elu>;

template <typename T>
using ExpOneDNNFunctor = OneDNNActivationFunc<T, dnnl::algorithm::eltwise_exp>;

template <typename T>
using GeluTanhOneDNNFunctor =
    OneDNNActivationFunc<T, dnnl::algorithm::eltwise_gelu_tanh>;

template <typename T>
using GeluErfOneDNNFunctor =
    OneDNNActivationFunc<T, dnnl::algorithm::eltwise_gelu_erf>;

template <typename T>
using HardSwishOneDNNFunctor =
    OneDNNActivationFunc<T, dnnl::algorithm::eltwise_hardswish>;

template <typename T>
using MishOneDNNFunctor =
    OneDNNActivationFunc<T, dnnl::algorithm::eltwise_mish>;

template <typename T>
using ReluOneDNNFunctor =
    OneDNNActivationFunc<T, dnnl::algorithm::eltwise_relu>;

template <typename T>
using Relu6OneDNNFunctor =
    OneDNNActivationFunc<T, dnnl::algorithm::eltwise_clip_v2>;

template <typename T>
using RoundOneDNNFunctor =
    OneDNNActivationFunc<T, dnnl::algorithm::eltwise_round>;

template <typename T>
using SigmoidOneDNNFunctor =
    OneDNNActivationFunc<T, dnnl::algorithm::eltwise_logistic>;

template <typename T>
using SqrtOneDNNFunctor =
    OneDNNActivationFunc<T, dnnl::algorithm::eltwise_sqrt>;

template <typename T>
using SwishOneDNNFunctor =
    OneDNNActivationFunc<T, dnnl::algorithm::eltwise_swish>;

template <typename T>
using TanhOneDNNFunctor =
    OneDNNActivationFunc<T, dnnl::algorithm::eltwise_tanh>;

DEFINE_ONEDNN_ACTIVATION_KERNEL(Abs, AbsOneDNNFunctor)
DEFINE_ONEDNN_ACTIVATION_KERNEL(Exp, ExpOneDNNFunctor)
DEFINE_ONEDNN_ACTIVATION_KERNEL(Relu, ReluOneDNNFunctor)
DEFINE_ONEDNN_ACTIVATION_KERNEL(Sigmoid, SigmoidOneDNNFunctor)
DEFINE_ONEDNN_ACTIVATION_KERNEL(Sqrt, SqrtOneDNNFunctor)
DEFINE_ONEDNN_ACTIVATION_KERNEL(Tanh, TanhOneDNNFunctor)

// round eltwise primitive doesn't support BF16, nor does it support grad
template <typename T, typename Context>
void RoundKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 const int decimals,
                 DenseTensor* out) {
  float ten_pow_decimals = std::pow(10, decimals);

  DenseTensor out1;
  DenseTensorMeta meta_out(x.dtype(), x.dims());
  out1.set_meta(meta_out);
  out1.set_lod(x.lod());
  out1.set_mem_desc(x.mem_desc());
  dev_ctx.template Alloc<T>(&out1);

  for (int i = 0; i < x.numel(); i++) {
    out1.data<T>()[i] = x.data<T>()[i] * ten_pow_decimals;
  }
  RoundOneDNNFunctor<T> functor;
  functor(dev_ctx, out1, 0, 0, out);

  for (int i = 0; i < x.numel(); i++) {
    out->data<T>()[i] = out->data<T>()[i] * (1 / ten_pow_decimals);
  }
}

DEFINE_ONEDNN_ACT_KERNEL_WITH_ONE_ATTRS(Elu, EluOneDNNFunctor, alpha)
DEFINE_ONEDNN_ACT_KERNEL_WITH_ONE_ATTRS(LeakyRelu, ReluOneDNNFunctor, alpha)
DEFINE_ONEDNN_ACT_KERNEL_WITH_ONE_ATTRS(Mish, MishOneDNNFunctor, threshold)

template <typename T, typename Context>
void HardSwishKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     DenseTensor* out) {
  HardSwishOneDNNFunctor<T> functor;
  functor(dev_ctx, x, 1.0 / 6.0, 1.0 / 2.0, out);
}

template <typename T, typename Context>
void GeluKernel(const Context& dev_ctx,
                const DenseTensor& x,
                bool approximate,
                DenseTensor* out) {
  if (approximate) {
    GeluTanhOneDNNFunctor<T> functor;
    functor(dev_ctx, x, 0, 0, out);
  } else {
    GeluErfOneDNNFunctor<T> functor;
    functor(dev_ctx, x, 0, 0, out);
  }
}

template <typename T, typename Context>
void Relu6Kernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 DenseTensor* out) {
  Relu6OneDNNFunctor<T> functor;
  functor(dev_ctx, x, 0, 6.0, out);
}

template <typename T, typename Context>
void SwishKernel(const Context& dev_ctx,
                 const DenseTensor& x,
                 DenseTensor* out) {
  SwishOneDNNFunctor<T> functor;
  functor(dev_ctx, x, 1.0, 0, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(round, OneDNN, ONEDNN, phi::RoundKernel, float) {}

#define PD_REGISTER_ACTIVATION_KERNEL(name, func) \
  PD_REGISTER_KERNEL(                             \
      name, OneDNN, ONEDNN, phi::func, float, phi::dtype::bfloat16) {}

PD_REGISTER_ACTIVATION_KERNEL(abs, AbsKernel)
PD_REGISTER_ACTIVATION_KERNEL(elu, EluKernel)
PD_REGISTER_ACTIVATION_KERNEL(exp, ExpKernel)
PD_REGISTER_ACTIVATION_KERNEL(gelu, GeluKernel)
PD_REGISTER_ACTIVATION_KERNEL(hardswish, HardSwishKernel)
PD_REGISTER_ACTIVATION_KERNEL(leaky_relu, LeakyReluKernel)
PD_REGISTER_ACTIVATION_KERNEL(mish, MishKernel)
PD_REGISTER_ACTIVATION_KERNEL(relu, ReluKernel)
PD_REGISTER_ACTIVATION_KERNEL(relu6, Relu6Kernel)
PD_REGISTER_ACTIVATION_KERNEL(sigmoid, SigmoidKernel)
PD_REGISTER_ACTIVATION_KERNEL(sqrt, SqrtKernel)
PD_REGISTER_ACTIVATION_KERNEL(swish, SwishKernel)
PD_REGISTER_ACTIVATION_KERNEL(tanh, TanhKernel)
