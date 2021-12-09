/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace {

using paddle::framework::Tensor;

template <typename T>
class ClipMKLDNNKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    this->RunKernel(ctx);
  }

  void RunKernel(const paddle::framework::ExecutionContext& ctx) const {
    const auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");

    paddle::platform::ActivationMKLDNNHandler<T> handler(
        dnnl::algorithm::eltwise_clip_v2, ctx, mkldnn_engine, ctx.GetPlace(),
        x);

    auto src_memory_p = handler.AcquireSrcMemory(x);
    auto dst_memory_p = handler.AcquireDstMemory(out);
    auto activation_p = handler.AcquireForwardPrimitive();

    auto& astream = paddle::platform::MKLDNNDeviceContext::tls().get_stream();
    activation_p->execute(astream, {{DNNL_ARG_FROM, *src_memory_p},
                                    {DNNL_ARG_TO, *dst_memory_p}});
    astream.wait();

    out->set_layout(paddle::framework::DataLayout::kMKLDNN);
    out->set_format(paddle::platform::GetMKLDNNFormat(*dst_memory_p));
  }
};

template <typename T>
class ClipGradMKLDNNKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    this->RunKernel(ctx);
  }

  void RunKernel(const paddle::framework::ExecutionContext& ctx) const {
    const auto& dev_ctx =
        ctx.template device_context<paddle::platform::MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto* x = ctx.Input<Tensor>("X");
    auto* dx = ctx.Output<Tensor>(paddle::framework::GradVarName("X"));
    auto* dout = ctx.Input<Tensor>(paddle::framework::GradVarName("Out"));

    paddle::platform::ActivationMKLDNNHandler<T> handler(
        dnnl::algorithm::eltwise_clip_v2, ctx, mkldnn_engine, ctx.GetPlace(), x,
        dout);

    auto src_memory_p = handler.AcquireBackwardSrcMemory(x);
    auto diff_dst_memory_p = handler.AcquireDiffDstMemory(dout);
    auto diff_src_memory_p = handler.AcquireDiffSrcMemory(dx);
    auto activation_backward_p = handler.AcquireBackwardPrimitive();

    auto& astream = paddle::platform::MKLDNNDeviceContext::tls().get_stream();
    activation_backward_p->execute(astream,
                                   {{DNNL_ARG_SRC, *src_memory_p},
                                    {DNNL_ARG_DIFF_DST, *diff_dst_memory_p},
                                    {DNNL_ARG_DIFF_SRC, *diff_src_memory_p}});
    astream.wait();

    dx->set_layout(paddle::framework::DataLayout::kMKLDNN);
    dx->set_format(paddle::platform::GetMKLDNNFormat(*diff_dst_memory_p));
  }
};

}  // anonymous namespace

REGISTER_OP_KERNEL(clip, MKLDNN, paddle::platform::CPUPlace,
                   ClipMKLDNNKernel<float>,
                   ClipMKLDNNKernel<paddle::platform::bfloat16>);

REGISTER_OP_KERNEL(clip_grad, MKLDNN, paddle::platform::CPUPlace,
                   ClipGradMKLDNNKernel<float>,
                   ClipGradMKLDNNKernel<paddle::platform::bfloat16>);
