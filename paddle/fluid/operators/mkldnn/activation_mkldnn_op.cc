/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/fluid/operators/activation_op.h"
#include "paddle/fluid/operators/mkldnn/softplus_mkldnn_op.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace phi {
class DenseTensor;
}  // namespace phi

namespace paddle {
namespace operators {

using dnnl::memory;
using dnnl::primitive;
using dnnl::stream;
using framework::DataLayout;
using framework::Tensor;
using platform::GetMKLDNNFormat;
using platform::MKLDNNDeviceContext;
using platform::to_void_cast;

template <typename Functor>
class MKLDNNActivationKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    Functor functor;
    functor(ctx);
  }
};

template <typename Functor>
class MKLDNNActivationGradKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    Functor functor;
    functor(ctx);
  }
};

template <typename T>
void eltwise_forward(const framework::ExecutionContext &ctx,
                     dnnl::algorithm algorithm) {
  PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()),
                    true,
                    paddle::platform::errors::PreconditionNotMet(
                        "Operator DNNL eletwise_forward must use CPUPlace"));
  auto &dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
  const auto &mkldnn_engine = dev_ctx.GetEngine();

  const auto *x = ctx.Input<Tensor>("X");
  auto *out = ctx.Output<Tensor>("Out");

  bool is_inplaced = x->IsSharedBufferWith(*out);

  platform::ActivationMKLDNNHandler<T> handler(
      algorithm, ctx, mkldnn_engine, ctx.GetPlace(), x);

  auto src_memory_p = handler.AcquireSrcMemory(x);
  std::shared_ptr<dnnl::memory> dst_memory_p = nullptr;
  if (is_inplaced) {
    dst_memory_p = src_memory_p;
    out->mutable_data<T>(ctx.GetPlace());
  } else {
    dst_memory_p = handler.AcquireDstMemory(out);
  }
  auto activation_p = handler.AcquireForwardPrimitive();

  auto &astream = paddle::platform::MKLDNNDeviceContext::tls().get_stream();
  activation_p->execute(
      astream, {{DNNL_ARG_FROM, *src_memory_p}, {DNNL_ARG_TO, *dst_memory_p}});
  astream.wait();

  out->set_mem_desc(dst_memory_p->get_desc());
}

template <typename T>
void eltwise_grad(const framework::ExecutionContext &ctx,
                  dnnl::algorithm algorithm) {
  auto &dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
  const auto &mkldnn_engine = dev_ctx.GetEngine();

  const auto *x = ctx.Input<Tensor>("X");
  const auto *dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
  auto *dx = ctx.Output<Tensor>(framework::GradVarName("X"));

  platform::ActivationMKLDNNHandler<T> handler(
      algorithm, ctx, mkldnn_engine, ctx.GetPlace(), x, dout);

  auto src_memory_p = handler.AcquireBackwardSrcMemory(x);
  auto diff_dst_memory_p = handler.AcquireDiffDstMemory(dout);
  auto diff_src_memory_p = handler.AcquireDiffSrcMemory(dx);
  auto activation_backward_p = handler.AcquireBackwardPrimitive();

  auto &astream = paddle::platform::MKLDNNDeviceContext::tls().get_stream();
  activation_backward_p->execute(astream,
                                 {{DNNL_ARG_SRC, *src_memory_p},
                                  {DNNL_ARG_DIFF_DST, *diff_dst_memory_p},
                                  {DNNL_ARG_DIFF_SRC, *diff_src_memory_p}});
  astream.wait();

  dx->set_mem_desc(diff_src_memory_p->get_desc());
}

template <typename T>
void eltwise_grad_use_out(const framework::ExecutionContext &ctx,
                          dnnl::algorithm algorithm) {
  auto &dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
  const auto &mkldnn_engine = dev_ctx.GetEngine();

  const auto *out = ctx.Input<Tensor>("Out");
  const auto *dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
  auto *dx = ctx.Output<Tensor>(framework::GradVarName("X"));

  platform::ActivationMKLDNNHandler<T> handler(
      algorithm, ctx, mkldnn_engine, ctx.GetPlace(), out, dout);

  auto dst_memory_p = handler.AcquireBackwardSrcMemory(out);
  auto diff_dst_memory_p = handler.AcquireDiffDstMemory(dout);
  auto diff_src_memory_p = handler.AcquireDiffSrcMemory(dx);
  auto activation_backward_p = handler.AcquireBackwardPrimitive();

  auto &astream = paddle::platform::MKLDNNDeviceContext::tls().get_stream();
  activation_backward_p->execute(astream,
                                 {{DNNL_ARG_DST, *dst_memory_p},
                                  {DNNL_ARG_DIFF_DST, *diff_dst_memory_p},
                                  {DNNL_ARG_DIFF_SRC, *diff_src_memory_p}});
  astream.wait();

  dx->set_mem_desc(diff_src_memory_p->get_desc());
}

template <typename T, dnnl::algorithm algorithm>
struct MKLDNNActivationGradFunc : public BaseActivationFunctor<T> {
  void operator()(const framework::ExecutionContext &ctx) const {
    eltwise_grad<T>(ctx, algorithm);
  }
};

template <typename T>
struct GeluMKLDNNFunctor : public BaseActivationFunctor<T> {
  void operator()(const framework::ExecutionContext &ctx) const {
    const bool approximate = ctx.Attr<bool>("approximate");
    if (approximate) {
      eltwise_forward<T>(ctx, dnnl::algorithm::eltwise_gelu_tanh);
    } else {
      eltwise_forward<T>(ctx, dnnl::algorithm::eltwise_gelu_erf);
    }
  }
};

template <typename T>
struct GeluMKLDNNGradFunctor : public BaseActivationFunctor<T> {
  void operator()(const framework::ExecutionContext &ctx) const {
    const bool approximate = ctx.Attr<bool>("approximate");
    if (approximate) {
      eltwise_grad<T>(ctx, dnnl::algorithm::eltwise_gelu_tanh);
    } else {
      eltwise_grad<T>(ctx, dnnl::algorithm::eltwise_gelu_erf);
    }
  }
};

template <typename T>
struct SoftplusMKLDNNFunctor : public BaseActivationFunctor<T> {
  void operator()(const framework::ExecutionContext &ctx) const {
    custom_softplus_eltwise_forward<T>(ctx);
  }
};

template <typename T>
using Relu6MKLDNNGradFunctor =
    MKLDNNActivationGradFunc<T, dnnl::algorithm::eltwise_bounded_relu>;

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

#define REGISTER_FWD_ACTIVATION_MKLDNN_KERNEL(act_type, functor) \
  REGISTER_OP_KERNEL(                                            \
      act_type,                                                  \
      MKLDNN,                                                    \
      ::paddle::platform::CPUPlace,                              \
      ops::MKLDNNActivationKernel<ops::functor<float>>,          \
      ops::MKLDNNActivationKernel<ops::functor<paddle::platform::bfloat16>>);

#define REGISTER_GRAD_ACTIVATION_MKLDNN_KERNEL(act_type, grad_functor) \
  REGISTER_OP_KERNEL(                                                  \
      act_type##_grad,                                                 \
      MKLDNN,                                                          \
      ::paddle::platform::CPUPlace,                                    \
      ops::MKLDNNActivationGradKernel<ops::grad_functor<float>>,       \
      ops::MKLDNNActivationGradKernel<                                 \
          ops::grad_functor<paddle::platform::bfloat16>>);

REGISTER_FWD_ACTIVATION_MKLDNN_KERNEL(softplus, SoftplusMKLDNNFunctor);
REGISTER_FWD_ACTIVATION_MKLDNN_KERNEL(gelu, GeluMKLDNNFunctor);
REGISTER_GRAD_ACTIVATION_MKLDNN_KERNEL(gelu, GeluMKLDNNGradFunctor);
REGISTER_GRAD_ACTIVATION_MKLDNN_KERNEL(relu6, Relu6MKLDNNGradFunctor);
