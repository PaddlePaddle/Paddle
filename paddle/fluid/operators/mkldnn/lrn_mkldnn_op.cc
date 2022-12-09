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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"

namespace paddle {
namespace operators {

using phi::OneDNNContext;

template <typename T>
class LRNOneDNNHandler
    : public phi::funcs::
          OneDNNHandlerNoCachingT<T, dnnl::lrn_forward, dnnl::lrn_backward> {
 public:
  LRNOneDNNHandler(const framework::ExecutionContext& ctx,
                   const dnnl::engine onednn_engine,
                   platform::Place cpu_place,
                   const phi::DenseTensor* input)

      : phi::funcs::
            OneDNNHandlerNoCachingT<T, dnnl::lrn_forward, dnnl::lrn_backward>(
                onednn_engine, cpu_place) {
    const int n = ctx.Attr<int>("n");
    // MKL-DNN implements LRN in a caffe way:
    // http://caffe.berkeleyvision.org/tutorial/layers/lrn.html
    // Where sum of squares is divided by size of normalization window
    // this is not the case for PaddlePaddle LRN.
    // Hence we need to compensate for this diffrence by
    // multipliing alpha by size of window(n)
    const float alpha = ctx.Attr<float>("alpha") * static_cast<float>(n);
    const float beta = ctx.Attr<float>("beta");
    const float k = ctx.Attr<float>("k");
    bool is_test = ctx.Attr<bool>("is_test");

    this->AcquireForwardPrimitiveDescriptor(
        is_test ? dnnl::prop_kind::forward_inference
                : dnnl::prop_kind::forward_training,
        dnnl::algorithm::lrn_across_channels,
        input->mem_desc(),
        n,
        alpha,
        beta,
        k);
  }

  LRNOneDNNHandler(const framework::ExecutionContext& ctx,
                   const dnnl::engine onednn_engine,
                   platform::Place cpu_place,
                   const phi::DenseTensor* in_x,
                   const phi::DenseTensor* out_grad,
                   phi::DenseTensor* in_x_grad)
      : phi::funcs::
            OneDNNHandlerNoCachingT<T, dnnl::lrn_forward, dnnl::lrn_backward>(
                onednn_engine, cpu_place) {
    PADDLE_ENFORCE_EQ(
        ctx.Attr<bool>("is_test"),
        false,
        platform::errors::PreconditionNotMet(
            "is_test attribute should be set to False in training phase."));

    const int n = ctx.Attr<int>("n");
    const float alpha = ctx.Attr<float>("alpha") * static_cast<float>(n);
    const float beta = ctx.Attr<float>("beta");
    const float k = ctx.Attr<float>("k");

    this->AcquireForwardPrimitiveDescriptor(
        dnnl::prop_kind::forward_training,
        dnnl::algorithm::lrn_across_channels,
        in_x->mem_desc(),
        n,
        alpha,
        beta,
        k);

    this->AcquireBackwardPrimitiveDescriptor(
        dnnl::algorithm::lrn_across_channels,
        in_x->mem_desc(),
        out_grad->mem_desc(),
        n,
        alpha,
        beta,
        k);
  }

  std::shared_ptr<dnnl::memory> AcquireWorkspaceMemory(
      phi::DenseTensor* workspace) {
    T* ptr = workspace->mutable_data<T>(
        this->place_, this->fwd_pd_->workspace_desc().get_size());
    return this->AcquireMemoryFromPrimitive(this->fwd_pd_->workspace_desc(),
                                            ptr);
  }

  std::shared_ptr<dnnl::memory> AcquireBackwardWorkspaceMemory(
      const phi::DenseTensor* workspace) {
    const T* workspace_data = workspace->data<T>();
    return this->AcquireMemoryFromPrimitive(
        this->fwd_pd_->workspace_desc(),
        phi::funcs::to_void_cast<T>(workspace_data));
  }
};

template <typename T>
class LRNMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    const bool is_float_type = std::is_same<T, float>::value;
    PADDLE_ENFORCE_EQ(
        is_float_type,
        true,
        platform::errors::PreconditionNotMet("DNNL LRN must use float data."));
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()),
                      true,
                      paddle::platform::errors::PreconditionNotMet(
                          "Operator DNNL LRN must use CPUPlace"));
    auto& dev_ctx = ctx.template device_context<OneDNNContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto x = ctx.Input<phi::DenseTensor>("X");
    auto out = ctx.Output<phi::DenseTensor>("Out");
    auto mid = ctx.Output<phi::DenseTensor>("MidOut");

    LRNOneDNNHandler<T> handler(ctx, onednn_engine, ctx.GetPlace(), x);

    auto src_memory = handler.AcquireSrcMemory(x);
    auto dst_memory = handler.AcquireDstMemory(out);

    auto lrn_p = handler.AcquireForwardPrimitive();

    auto workspace_memory = handler.AcquireWorkspaceMemory(mid);
    mid->set_layout(phi::DataLayout::ONEDNN);

    auto& astream = OneDNNContext::tls().get_stream();
    if (!workspace_memory->get_desc().is_zero()) {
      mid->set_mem_desc(workspace_memory->get_desc());
      lrn_p->execute(astream,
                     {{DNNL_ARG_SRC, *src_memory},
                      {DNNL_ARG_DST, *dst_memory},
                      {DNNL_ARG_WORKSPACE, *workspace_memory}});
    } else {
      lrn_p->execute(
          astream, {{DNNL_ARG_SRC, *src_memory}, {DNNL_ARG_DST, *dst_memory}});
    }
    astream.wait();

    out->set_mem_desc(dst_memory->get_desc());
  }
};

template <typename T>
class LRNMKLDNNGradOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    const bool is_float_type = std::is_same<T, float>::value;
    PADDLE_ENFORCE_EQ(is_float_type,
                      true,
                      platform::errors::PreconditionNotMet(
                          "DNNL LRN GradOpKernel must use float data."));
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()),
                      true,
                      paddle::platform::errors::PreconditionNotMet(
                          "Operator DNNL LRNGrad must use CPUPlace"));

    auto in_x = ctx.Input<phi::DenseTensor>("X");
    auto mid = ctx.Input<phi::DenseTensor>("MidOut");

    auto out_grad = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto in_x_grad = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));

    auto& dev_ctx = ctx.template device_context<OneDNNContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    LRNOneDNNHandler<T> handler(
        ctx, onednn_engine, ctx.GetPlace(), in_x, out_grad, in_x_grad);

    auto src_memory = handler.AcquireSrcMemory(in_x);
    auto workspace = handler.AcquireBackwardWorkspaceMemory(mid);
    auto diff_dst_memory = handler.AcquireDiffDstMemory(out_grad);
    auto diff_src_memory = handler.AcquireDiffSrcMemory(in_x_grad);

    auto lrn_bwd = handler.AcquireBackwardPrimitive();

    auto& astream = OneDNNContext::tls().get_stream();
    lrn_bwd->execute(astream,
                     {{DNNL_ARG_SRC, *src_memory},
                      {DNNL_ARG_DIFF_DST, *diff_dst_memory},
                      {DNNL_ARG_DIFF_SRC, *diff_src_memory},
                      {DNNL_ARG_WORKSPACE, *workspace}});
    astream.wait();

    in_x_grad->set_mem_desc(diff_src_memory->get_desc());
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(lrn, MKLDNN, phi::CPUPlace, ops::LRNMKLDNNOpKernel<float>);
REGISTER_OP_KERNEL(lrn_grad,
                   MKLDNN,
                   phi::CPUPlace,
                   ops::LRNMKLDNNGradOpKernel<float>);
