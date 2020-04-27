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
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using framework::Tensor;
using mkldnn::memory;
using mkldnn::primitive;
using mkldnn::stream;
using platform::GetMKLDNNFormat;
using platform::MKLDNNDeviceContext;
using platform::to_void_cast;

template <typename Functor>
class MKLDNNActivationKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *x = ctx.Input<Tensor>("X");
    PADDLE_ENFORCE_EQ(x->layout(), DataLayout::kMKLDNN,
                      "Wrong layout set for X tensor");
    PADDLE_ENFORCE_NE(x->format(), MKLDNNMemoryFormat::undef,
                      "Wrong format set for X tensor");

    Functor functor;
    functor(ctx);
  }
};

template <typename Functor>
class MKLDNNActivationGradKernel
    : public framework::OpKernel<typename Functor::ELEMENT_TYPE> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *diff_y = ctx.Input<Tensor>(framework::GradVarName("Out"));
    PADDLE_ENFORCE_EQ(diff_y->layout(), DataLayout::kMKLDNN,
                      "Wrong layout set for Input OutGrad tensor");
    PADDLE_ENFORCE_NE(diff_y->format(), MKLDNNMemoryFormat::undef,
                      "Wrong format set for Input OutGrad tensor");

    Functor functor;
    functor(ctx);
  }
};

template <typename T>
void eltwise_forward(const framework::ExecutionContext &ctx,
                     mkldnn::algorithm algorithm) {
  PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                 "It must use CPUPlace.");
  auto &dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();

  const auto *x = ctx.Input<Tensor>("X");
  auto *y = ctx.Output<Tensor>("Out");

  T alpha = ctx.HasAttr("alpha") ? ctx.Attr<T>("alpha") : 0;
  T beta = ctx.HasAttr("beta") ? ctx.Attr<T>("beta") : 0;

  // paddle uses beta but mkldnn uses alpha for swish
  if (algorithm == mkldnn::algorithm::eltwise_swish) {
    std::swap(alpha, beta);
  }

  PADDLE_ENFORCE(
      x->dims().size() == 2 || x->dims().size() == 3 || x->dims().size() == 4,
      "Input dim must be with 2, 3 or 4");

  auto src_tz = framework::vectorize<int64_t>(x->dims());

  auto src_format = src_tz.size() == 2 ? MKLDNNMemoryFormat::nc : x->format();

  platform::ActivationMKLDNNHandler<T> handler(
      src_tz, algorithm, alpha, beta, src_format, dev_ctx, ctx.GetPlace(),
      ctx.InputName("X"));

  auto src_memory_p = handler.AcquireSrcMemory(x);
  auto dst_memory_p =
      x->IsSharedBufferWith(*y) ? src_memory_p : handler.AcquireDstMemory(y);
  auto activation_p = handler.AcquireForwardPrimitive();

  mkldnn::stream astream(dev_ctx.GetEngine());
  activation_p->execute(astream, {{MKLDNN_ARG_FROM, *src_memory_p},
                                  {MKLDNN_ARG_TO, *dst_memory_p}});
  astream.wait();

  y->set_layout(DataLayout::kMKLDNN);
  y->set_format(GetMKLDNNFormat(*dst_memory_p));
}

template <typename T>
void eltwise_grad(const framework::ExecutionContext &ctx,
                  mkldnn::algorithm algorithm) {
  auto &dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();

  const auto *x = ctx.Input<Tensor>("X");
  const auto *diff_y = ctx.Input<Tensor>(framework::GradVarName("Out"));
  auto *diff_x = ctx.Output<Tensor>(framework::GradVarName("X"));

  T alpha = ctx.HasAttr("alpha") ? ctx.Attr<T>("alpha") : 0;
  T beta = ctx.HasAttr("beta") ? ctx.Attr<T>("beta") : 0;

  // paddle uses beta but mkldnn uses alpha for swish
  if (algorithm == mkldnn::algorithm::eltwise_swish) {
    std::swap(alpha, beta);
  }

  auto diff_dst_tz = framework::vectorize<int64_t>(diff_y->dims());

  // diff_dst and src dims should be the same
  auto src_format =
      diff_dst_tz.size() == 2 ? MKLDNNMemoryFormat::nc : x->format();

  auto diff_y_format =
      diff_dst_tz.size() == 2 ? MKLDNNMemoryFormat::nc : diff_y->format();

  platform::ActivationMKLDNNHandler<T> handler(
      diff_dst_tz, algorithm, alpha, beta, src_format, diff_y_format, dev_ctx,
      ctx.GetPlace(), ctx.InputName("X"));

  auto src_memory_p = handler.AcquireBackwardSrcMemory(x);
  auto diff_dst_memory_p = handler.AcquireDiffDstMemory(diff_y);
  auto diff_src_memory_p = handler.AcquireDiffSrcMemory(diff_x);
  auto activation_backward_p = handler.AcquireBackwardPrimitive();

  mkldnn::stream astream(dev_ctx.GetEngine());
  activation_backward_p->execute(astream,
                                 {{MKLDNN_ARG_SRC, *src_memory_p},
                                  {MKLDNN_ARG_DIFF_DST, *diff_dst_memory_p},
                                  {MKLDNN_ARG_DIFF_SRC, *diff_src_memory_p}});
  astream.wait();

  diff_x->set_layout(DataLayout::kMKLDNN);
  diff_x->set_format(GetMKLDNNFormat(*diff_src_memory_p));
}

template <typename T, mkldnn::algorithm algorithm>
struct MKLDNNActivationFunc : public BaseActivationFunctor<T> {
  void operator()(const framework::ExecutionContext &ctx) const {
    eltwise_forward<T>(ctx, algorithm);
  }
};

template <typename T, mkldnn::algorithm algorithm>
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
      eltwise_forward<T>(ctx, mkldnn::algorithm::eltwise_gelu_tanh);
    } else {
      eltwise_forward<T>(ctx, mkldnn::algorithm::eltwise_gelu_erf);
    }
  }
};

template <typename T>
struct GeluMKLDNNGradFunctor : public BaseActivationFunctor<T> {
  void operator()(const framework::ExecutionContext &ctx) const {
    const bool approximate = ctx.Attr<bool>("approximate");
    if (approximate) {
      eltwise_grad<T>(ctx, mkldnn::algorithm::eltwise_gelu_tanh);
    } else {
      eltwise_grad<T>(ctx, mkldnn::algorithm::eltwise_gelu_erf);
    }
  }
};

template <typename T>
using ReluMKLDNNFunctor =
    MKLDNNActivationFunc<T, mkldnn::algorithm::eltwise_relu>;

template <typename T>
using SwishMKLDNNFunctor =
    MKLDNNActivationFunc<T, mkldnn::algorithm::eltwise_swish>;

template <typename T>
using TanhMKLDNNFunctor =
    MKLDNNActivationFunc<T, mkldnn::algorithm::eltwise_tanh>;

template <typename T>
using SqrtMKLDNNFunctor =
    MKLDNNActivationFunc<T, mkldnn::algorithm::eltwise_sqrt>;

template <typename T>
using AbsMKLDNNFunctor =
    MKLDNNActivationFunc<T, mkldnn::algorithm::eltwise_abs>;

template <typename T>
using ReluMKLDNNGradFunctor =
    MKLDNNActivationGradFunc<T, mkldnn::algorithm::eltwise_relu>;

template <typename T>
using SwishMKLDNNGradFunctor =
    MKLDNNActivationGradFunc<T, mkldnn::algorithm::eltwise_swish>;

template <typename T>
using TanhMKLDNNGradFunctor =
    MKLDNNActivationGradFunc<T, mkldnn::algorithm::eltwise_tanh>;

template <typename T>
using SqrtMKLDNNGradFunctor =
    MKLDNNActivationGradFunc<T, mkldnn::algorithm::eltwise_sqrt>;

template <typename T>
using AbsMKLDNNGradFunctor =
    MKLDNNActivationGradFunc<T, mkldnn::algorithm::eltwise_abs>;
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

#define REGISTER_ACTIVATION_MKLDNN_KERNEL(act_type, functor, grad_functor) \
  REGISTER_OP_KERNEL(act_type, MKLDNN, ::paddle::platform::CPUPlace,       \
                     ops::MKLDNNActivationKernel<ops::functor<float>>);    \
  REGISTER_OP_KERNEL(                                                      \
      act_type##_grad, MKLDNN, ::paddle::platform::CPUPlace,               \
      ops::MKLDNNActivationGradKernel<ops::grad_functor<float>>);

#define FOR_EACH_MKLDNN_KERNEL_FUNCTOR(__macro)                  \
  __macro(relu, ReluMKLDNNFunctor, ReluMKLDNNGradFunctor);       \
  __macro(leaky_relu, ReluMKLDNNFunctor, ReluMKLDNNGradFunctor); \
  __macro(gelu, GeluMKLDNNFunctor, GeluMKLDNNGradFunctor);       \
  __macro(swish, SwishMKLDNNFunctor, SwishMKLDNNGradFunctor);    \
  __macro(tanh, TanhMKLDNNFunctor, TanhMKLDNNGradFunctor);       \
  __macro(sqrt, SqrtMKLDNNFunctor, SqrtMKLDNNGradFunctor);       \
  __macro(abs, AbsMKLDNNFunctor, AbsMKLDNNGradFunctor);

FOR_EACH_MKLDNN_KERNEL_FUNCTOR(REGISTER_ACTIVATION_MKLDNN_KERNEL);
