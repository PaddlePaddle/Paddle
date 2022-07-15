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

#include "paddle/phi/backends/onednn/onednn_context.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/activation_functor.h"
#include "paddle/phi/kernels/funcs/onednn/mkldnn_reuse.h"

namespace phi {

using dnnl::memory;
using dnnl::primitive;
using dnnl::stream;

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
void eltwise_forward(const OneDNNContext& dev_ctx,
                     const DenseTensor& x,
                     float alpha,
                     float beta,
                     DenseTensor* out,
                     dnnl::algorithm algorithm) {
  PADDLE_ENFORCE_EQ(paddle::platform::is_cpu_place(dev_ctx.GetPlace()),
                    true,
                    phi::errors::PreconditionNotMet(
                        "Operator DNNL eletwise_forward must use ONEDNNPlace"));
  const auto& mkldnn_engine = dev_ctx.GetEngine();

  bool is_inplaced = x.IsSharedBufferWith(*out);

  funcs::ActivationMKLDNNHandler<T> handler(
      algorithm, alpha, beta, mkldnn_engine, dev_ctx.GetPlace(), &x);

  auto src_memory_p = handler.AcquireSrcMemory(&x);
  std::shared_ptr<dnnl::memory> dst_memory_p = nullptr;
  if (is_inplaced) {
    dst_memory_p = src_memory_p;
    out->mutable_data<T>(dev_ctx.GetPlace());
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
struct MKLDNNActivationFunc : public funcs::BaseActivationFunctor<T> {
  void operator()(const OneDNNContext& dev_ctx,
                  const DenseTensor& x,
                  float alpha,
                  float beta,
                  DenseTensor* out) const {
    eltwise_forward<T>(dev_ctx, x, alpha, beta, out, algorithm);
  }
};

template <typename T>
using ReluMKLDNNFunctor =
    MKLDNNActivationFunc<T, dnnl::algorithm::eltwise_relu>;

template <typename T>
using SwishMKLDNNFunctor =
    MKLDNNActivationFunc<T, dnnl::algorithm::eltwise_swish>;

template <typename T>
using HardSwishMKLDNNFunctor =
    MKLDNNActivationFunc<T, dnnl::algorithm::eltwise_hardswish>;

template <typename T>
using MishMKLDNNFunctor =
    MKLDNNActivationFunc<T, dnnl::algorithm::eltwise_mish>;

template <typename T>
using SigmoidMKLDNNFunctor =
    MKLDNNActivationFunc<T, dnnl::algorithm::eltwise_logistic>;

template <typename T>
using TanhMKLDNNFunctor =
    MKLDNNActivationFunc<T, dnnl::algorithm::eltwise_tanh>;

template <typename T>
using SqrtMKLDNNFunctor =
    MKLDNNActivationFunc<T, dnnl::algorithm::eltwise_sqrt>;

template <typename T>
using EluMKLDNNFunctor = MKLDNNActivationFunc<T, dnnl::algorithm::eltwise_elu>;

template <typename T>
using ExpMKLDNNFunctor = MKLDNNActivationFunc<T, dnnl::algorithm::eltwise_exp>;

template <typename T>
using RoundMKLDNNFunctor =
    MKLDNNActivationFunc<T, dnnl::algorithm::eltwise_round>;

DEFINE_ONEDNN_ACTIVATION_KERNEL(Relu, ReluMKLDNNFunctor)
DEFINE_ONEDNN_ACTIVATION_KERNEL(Tanh, TanhMKLDNNFunctor)
DEFINE_ONEDNN_ACTIVATION_KERNEL(Exp, ExpMKLDNNFunctor)
DEFINE_ONEDNN_ACTIVATION_KERNEL(Sqrt, SqrtMKLDNNFunctor)
DEFINE_ONEDNN_ACTIVATION_KERNEL(Sigmoid, SigmoidMKLDNNFunctor)
// round eltwise primitive doesn't support BF16, nor does it support grad
DEFINE_ONEDNN_ACTIVATION_KERNEL(Round, RoundMKLDNNFunctor)

DEFINE_ONEDNN_ACT_KERNEL_WITH_ONE_ATTRS(LeakyRelu, ReluMKLDNNFunctor, alpha)
DEFINE_ONEDNN_ACT_KERNEL_WITH_ONE_ATTRS(Mish, MishMKLDNNFunctor, threshold)
DEFINE_ONEDNN_ACT_KERNEL_WITH_ONE_ATTRS(Elu, EluMKLDNNFunctor, alpha)
DEFINE_ONEDNN_ACT_KERNEL_WITH_ONE_ATTRS(Swish, SwishMKLDNNFunctor, beta)

template <typename T, typename Context>
void HardSwishKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     float threshold,
                     float scale,
                     float offset,
                     DenseTensor* out) {
  HardSwishMKLDNNFunctor<T> functor;
  functor(dev_ctx, x, threshold, 0, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(round, OneDNN, ALL_LAYOUT, phi::RoundKernel, float) {}

#define PD_REGISTER_ACTIVATION_KERNEL(name, func) \
  PD_REGISTER_KERNEL(                             \
      name, OneDNN, ALL_LAYOUT, phi::func, float, phi::dtype::bfloat16) {}

PD_REGISTER_ACTIVATION_KERNEL(elu, EluKernel)
PD_REGISTER_ACTIVATION_KERNEL(exp, ExpKernel)
PD_REGISTER_ACTIVATION_KERNEL(hard_swish, HardSwishKernel)
PD_REGISTER_ACTIVATION_KERNEL(leaky_relu, LeakyReluKernel)
PD_REGISTER_ACTIVATION_KERNEL(mish, MishKernel)
PD_REGISTER_ACTIVATION_KERNEL(sigmoid, SigmoidKernel)
PD_REGISTER_ACTIVATION_KERNEL(sqrt, SqrtKernel)
PD_REGISTER_ACTIVATION_KERNEL(swish, SwishKernel)
PD_REGISTER_ACTIVATION_KERNEL(tanh, TanhKernel)
PD_REGISTER_ACTIVATION_KERNEL(relu, ReluKernel)
