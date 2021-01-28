/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/pool_op.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using mkldnn::memory;
using mkldnn::pooling_backward;
using mkldnn::pooling_forward;
using mkldnn::primitive;
using mkldnn::reorder;
using mkldnn::stream;
using platform::to_void_cast;

template <typename T>
class PoolMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()), true,
                      paddle::platform::errors::PreconditionNotMet(
                          "Operator DNNL Pool must use CPUPlace"));
    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();

    const Tensor* input = ctx.Input<Tensor>("X");
    Tensor* output = ctx.Output<Tensor>("Out");

    platform::PoolingMKLDNNHandler<T> handler(
        ctx, dev_ctx, ctx.GetPlace(), input, output, ctx.OutputName("Out"));

    auto src_memory = handler.AcquireSrcMemory(input);
    auto dst_memory = handler.AcquireDstMemory(output);

    auto pool_p = handler.AcquireForwardPrimitive();

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    if ((ctx.Attr<bool>("is_test") == false) &&
        (ctx.Attr<std::string>("pooling_type") == "max")) {
      // Training
      auto workspace_memory = handler.AcquireWorkspaceMemory();
      pool_p->execute(astream, {{MKLDNN_ARG_SRC, *src_memory},
                                {MKLDNN_ARG_DST, *dst_memory},
                                {MKLDNN_ARG_WORKSPACE, *workspace_memory}});
    } else {
      // Inference
      pool_p->execute(astream, {{MKLDNN_ARG_SRC, *src_memory},
                                {MKLDNN_ARG_DST, *dst_memory}});
    }
    astream.wait();

    output->set_layout(DataLayout::kMKLDNN);
    output->set_format(platform::GetMKLDNNFormat(*dst_memory));
  }
};

template <typename T>
class PoolMKLDNNGradOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(platform::is_cpu_place(ctx.GetPlace()), true,
                      paddle::platform::errors::PreconditionNotMet(
                          "Operator DNNL PoolGrad must use CPUPlace"));
    const Tensor* in_x = ctx.Input<Tensor>("X");
    const Tensor* out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    Tensor* in_x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();

    platform::PoolingMKLDNNHandler<T> handler(ctx, dev_ctx, ctx.GetPlace(),
                                              in_x, out_grad, in_x_grad,
                                              ctx.InputName("Out"));

    auto diff_dst_memory = handler.AcquireDiffDstMemory(out_grad);
    auto diff_src_memory = handler.AcquireDiffSrcMemory(in_x_grad);

    auto pool_bwd_p = handler.AcquireBackwardPrimitive();

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    if (ctx.Attr<std::string>("pooling_type") == "max") {
      // Max - pooling needs Workspace
      auto workspace_memory = handler.AcquireWorkspaceMemory();
      pool_bwd_p->execute(astream, {{MKLDNN_ARG_DIFF_SRC, *diff_src_memory},
                                    {MKLDNN_ARG_DIFF_DST, *diff_dst_memory},
                                    {MKLDNN_ARG_WORKSPACE, *workspace_memory}});
    } else {
      // Average Pooling
      pool_bwd_p->execute(astream, {{MKLDNN_ARG_DIFF_SRC, *diff_src_memory},
                                    {MKLDNN_ARG_DIFF_DST, *diff_dst_memory}});
    }
    astream.wait();

    in_x_grad->set_layout(DataLayout::kMKLDNN);
    in_x_grad->set_format(platform::GetMKLDNNFormat(*diff_src_memory));
  }  // Compute()
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(pool2d, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::PoolMKLDNNOpKernel<float>,
                   ops::PoolMKLDNNOpKernel<int8_t>,
                   ops::PoolMKLDNNOpKernel<uint8_t>,
                   ops::PoolMKLDNNOpKernel<paddle::platform::bfloat16>);

REGISTER_OP_KERNEL(pool2d_grad, MKLDNN, ::paddle::platform::CPUPlace,
                   ops::PoolMKLDNNGradOpKernel<float>);
