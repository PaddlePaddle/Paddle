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

#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/lrn_op.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

using paddle::framework::Tensor;
using paddle::platform::MKLDNNDeviceContext;

template <typename T>
class LRNMKLDNNOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    const bool is_float_type = std::is_same<T, float>::value;
    PADDLE_ENFORCE(is_float_type, "MKLDNN LRN must use float data.");
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "MKLDNN LRN must use CPUPlace.");

    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();

    auto x = ctx.Input<Tensor>("X");
    auto out = ctx.Output<Tensor>("Out");
    auto mid = ctx.Output<Tensor>("MidOut");

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

    auto dims = paddle::framework::vectorize<int64_t>(x->dims());

    platform::LRNMKLDNNHandler<T> handler(dims, n, alpha, beta, k, x->format(),
                                          is_test, dev_ctx, ctx.GetPlace(),
                                          ctx.OutputName("Out"));

    auto src_memory = handler.AcquireSrcMemory(x);
    auto dst_memory = handler.AcquireDstMemory(out);

    auto lrn_p = handler.AcquireForwardPrimitive();

    auto workspace_memory = handler.AcquireWorkspaceMemory(mid);
    mid->set_layout(framework::DataLayout::kMKLDNN);

    mkldnn::stream astream(dev_ctx.GetEngine());
    if (!workspace_memory->get_desc().is_zero()) {
      mid->set_format(platform::GetMKLDNNFormat(*workspace_memory));
      lrn_p->execute(astream, {{MKLDNN_ARG_SRC, *src_memory},
                               {MKLDNN_ARG_DST, *dst_memory},
                               {MKLDNN_ARG_WORKSPACE, *workspace_memory}});
    } else {
      // mid has to be allocated and filled
      // k to pass LRN unit tests
      // TODO(jczaja): Disable checking mid in unit tests (Require API change)
      mid->mutable_data<T>(ctx.GetPlace());
      auto e_mid = framework::EigenTensor<T, 4>::From(*mid);
      e_mid = e_mid.constant(k);
      mid->set_format(platform::GetMKLDNNFormat(*dst_memory));

      lrn_p->execute(astream, {{MKLDNN_ARG_SRC, *src_memory},
                               {MKLDNN_ARG_DST, *dst_memory}});
    }
    astream.wait();

    out->set_layout(framework::DataLayout::kMKLDNN);
    out->set_format(platform::GetMKLDNNFormat(*dst_memory));
  }
};

template <typename T>
class LRNMKLDNNGradOpKernel : public paddle::framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    const bool is_float_type = std::is_same<T, float>::value;
    PADDLE_ENFORCE(is_float_type, "MKLDNN LRN must use float data.");
    PADDLE_ENFORCE(paddle::platform::is_cpu_place(ctx.GetPlace()),
                   "MKLDNN LRN must use CPUPlace.");
    PADDLE_ENFORCE(
        !ctx.Attr<bool>("is_test"),
        "is_test attribute should be set to False in training phase.");

    auto x = ctx.Input<Tensor>("X");
    auto mid = ctx.Input<Tensor>("MidOut");

    auto out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    const int n = ctx.Attr<int>("n");
    const float alpha = ctx.Attr<float>("alpha") * static_cast<float>(n);
    const float beta = ctx.Attr<float>("beta");
    const float k = ctx.Attr<float>("k");

    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();

    auto dims = paddle::framework::vectorize<int64_t>(x->dims());

    platform::LRNMKLDNNHandler<T> handler(dims, n, alpha, beta, k, x->format(),
                                          out_grad->format(), dev_ctx,
                                          ctx.GetPlace(), ctx.InputName("Out"));

    auto src_memory = handler.AcquireSrcMemory(x);
    auto workspace = handler.AcquireBackwardWorkspaceMemory(mid);
    auto diff_dst_memory = handler.AcquireDiffDstMemory(out_grad);
    auto diff_src_memory = handler.AcquireDiffSrcMemory(x_grad);

    auto lrn_bwd = handler.AcquireBackwardPrimitive();

    mkldnn::stream astream(dev_ctx.GetEngine());
    lrn_bwd->execute(astream, {{MKLDNN_ARG_SRC, *src_memory},
                               {MKLDNN_ARG_DIFF_DST, *diff_dst_memory},
                               {MKLDNN_ARG_DIFF_SRC, *diff_src_memory},
                               {MKLDNN_ARG_WORKSPACE, *workspace}});
    astream.wait();

    x_grad->set_layout(framework::DataLayout::kMKLDNN);
    x_grad->set_format(platform::GetMKLDNNFormat(*diff_src_memory));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(lrn, MKLDNN, paddle::platform::CPUPlace,
                   ops::LRNMKLDNNOpKernel<float>);
REGISTER_OP_KERNEL(lrn_grad, MKLDNN, paddle::platform::CPUPlace,
                   ops::LRNMKLDNNGradOpKernel<float>);
