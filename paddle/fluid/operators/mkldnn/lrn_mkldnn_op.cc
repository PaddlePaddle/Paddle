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
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto x = ctx.Input<Tensor>("X");
    auto out = ctx.Output<Tensor>("Out");
    auto mid = ctx.Output<Tensor>("MidOut");

    auto input_data = x->data<T>();
    auto output_data = out->mutable_data<T>(ctx.GetPlace());
    mid->mutable_data<T>(ctx.GetPlace());

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

    auto e_mid = framework::EigenTensor<T, 4>::From(*mid);
    e_mid = e_mid.constant(k);

    auto dims = paddle::framework::vectorize2int(x->dims());

    // Format and dims are assumed to be the same for dst and src
    auto md = paddle::platform::MKLDNNMemDesc(
        dims, platform::MKLDNNGetDataType<T>(), x->format());

    const std::string key = platform::LRNMKLDNNHandler::GetHash(
        dims, n, alpha, beta, k, x->format(), ctx.op().Output("Out"));

    platform::LRNMKLDNNHandler handler(ctx.Attr<bool>("is_test"), dev_ctx,
                                       mkldnn_engine, key);
    auto src_memory =
        handler.AcquireSrcMemory(md, platform::to_void_cast<T>(input_data));

    // TODO(jczaja): Hide getting PD inside of handler for all Acquire API
    handler.AcquireLRNPrimitiveDescriptor(md, n, alpha, beta, k);

    auto dst_memory =
        handler.AcquireDstMemory(md, platform::to_void_cast<T>(output_data));

    auto lrn_p = handler.AcquireLRN(dst_memory, src_memory);

    std::vector<mkldnn::primitive> pipeline = {*lrn_p};
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();

    auto output_format =
        (mkldnn::memory::format)dst_memory->get_primitive_desc()
            .desc()
            .data.format;

    out->set_layout(framework::DataLayout::kMKLDNN);
    out->set_format(output_format);
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

    auto out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    const int n = ctx.Attr<int>("n");
    const float alpha = ctx.Attr<float>("alpha") * static_cast<float>(n);
    const float beta = ctx.Attr<float>("beta");
    const float k = ctx.Attr<float>("k");

    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();

    auto x_grad_data = x_grad->mutable_data<T>(ctx.GetPlace());
    auto out_grad_data = out_grad->data<T>();

    auto dims = paddle::framework::vectorize2int(x->dims());

    const std::string key = platform::LRNMKLDNNHandler::GetHash(
        dims, n, alpha, beta, k, x->format(), ctx.op().Input("Out"));

    platform::LRNMKLDNNHandler handler(false, dev_ctx, mkldnn_engine, key);

    auto src_md = paddle::platform::MKLDNNMemDesc(
        dims, platform::MKLDNNGetDataType<T>(), x->format());

    // diff_dst and diff_src layouts are assumed to be the same
    auto diff_md = paddle::platform::MKLDNNMemDesc(
        dims, platform::MKLDNNGetDataType<T>(), out_grad->format());

    auto workspace = handler.AcquireWorkspaceMemory();

    auto diff_dst_memory = handler.AcquireDiffDstMemory(
        diff_md, platform::to_void_cast<T>(out_grad_data));

    auto diff_src_memory = handler.AcquireDiffSrcMemory(
        diff_md, platform::to_void_cast<T>(x_grad_data));

    auto src_memory = handler.AcquireSrcMemory(
        src_md, platform::to_void_cast<T>(x->data<T>()));

    // TODO(jczaja): Hide this call inside Handler
    handler.AcquireLRNBackwardPrimitiveDescriptor(src_md, diff_md, n, alpha,
                                                  beta, k);

    auto lrn_bwd = handler.AcquireLRNBackward(src_memory, diff_dst_memory,
                                              workspace, diff_src_memory);

    std::vector<mkldnn::primitive> pipeline = {*lrn_bwd};
    mkldnn::stream(mkldnn::stream::kind::eager).submit(pipeline).wait();

    auto output_format =
        (mkldnn::memory::format)diff_src_memory->get_primitive_desc()
            .desc()
            .data.format;

    x_grad->set_layout(framework::DataLayout::kMKLDNN);
    x_grad->set_format(output_format);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_KERNEL(lrn, MKLDNN, paddle::platform::CPUPlace,
                   ops::LRNMKLDNNOpKernel<float>);
REGISTER_OP_KERNEL(lrn_grad, MKLDNN, paddle::platform::CPUPlace,
                   ops::LRNMKLDNNGradOpKernel<float>);
