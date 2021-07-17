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

#include "paddle/fluid/operators/squeeze_op.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace paddle {
namespace operators {

framework::DDim GetOutputShape(const std::vector<int> squeeze_dims,
                               const framework::DDim& in_dims, bool is_runtime);

using paddle::framework::LoDTensor;
using paddle::framework::Tensor;
using platform::to_void_cast;

template <typename T>
class SqueezeMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    RunKernel(ctx);
  }

  void RunKernel(const framework::ExecutionContext& ctx) const {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto* x = ctx.Input<LoDTensor>("X");
    auto* out = ctx.Output<LoDTensor>("Out");

    auto& axes = ctx.Attr<std::vector<int>>("axes");
    auto x_dims = x->dims();
    auto x_vec_dims = framework::vectorize(x_dims);
    auto out_dims = GetOutputShape(axes, x_dims, true);

    mkldnn::memory::data_type x_type = framework::ToMKLDNNDataType(x->type());
    std::string key = platform::CreateKey(dev_ctx, x_vec_dims, x->format(),
                                          x->format(), x_type);
    platform::ReorderMKLDNNHandler reorder_handler(
        x_vec_dims, x->type(), x_type, dev_ctx, onednn_engine, key);

    auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
        x->format(), platform::to_void_cast(x->data<T>()));
    auto reorder_dst_memory_p =
        reorder_handler.AcquireDstMemory(out, x->format(), ctx.GetPlace());
    auto reorder_p = reorder_handler.AcquireReorder(reorder_src_memory_p,
                                                    reorder_dst_memory_p);

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);

    astream.wait();

    out->Resize(out_dims);
    out->set_layout(framework::DataLayout::kMKLDNN);
    out->set_format(getPlainFormatTag(out));
  }

 protected:
  mkldnn::memory::format_tag getPlainFormatTag(const Tensor* tensor) const {
    auto tensor_dims_size = tensor->dims().size();
    PADDLE_ENFORCE_EQ(
        tensor_dims_size <= 6 && tensor_dims_size >= 1, true,
        platform::errors::InvalidArgument(
            "Dims for squeeze_grad oneDNN op must be in range <1, 6>"));

    switch (tensor_dims_size) {
      case 1:
        return mkldnn::memory::format_tag::a;
      case 2:
        return mkldnn::memory::format_tag::ab;
      case 3:
        return mkldnn::memory::format_tag::abc;
      case 4:
        return mkldnn::memory::format_tag::abcd;
      case 5:
        return mkldnn::memory::format_tag::abcde;
    }

    return mkldnn::memory::format_tag::abcdef;
  }
};

template <typename T>
class SqueezeGradMKLDNNKernel : public SqueezeMKLDNNKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    RunKernel(ctx);
  }

  void RunKernel(const framework::ExecutionContext& ctx) const {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto* dout = ctx.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<LoDTensor>(framework::GradVarName("X"));

    dx->set_format(this->getPlainFormatTag(dx));

    framework::DDim x_dims;
    if (ctx.Type() == "squeeze_grad") {
      x_dims = ctx.Input<LoDTensor>("X")->dims();
    } else {
      auto xshape_dims = ctx.Input<framework::LoDTensor>("XShape")->dims();
      x_dims = framework::slice_ddim(xshape_dims, 1, xshape_dims.size());
    }
    auto x_vec_dims = framework::vectorize(x_dims);

    mkldnn::memory::data_type dout_type =
        framework::ToMKLDNNDataType(dout->type());
    std::string key = platform::CreateKey(dev_ctx, x_vec_dims, dout->format(),
                                          dout->format(), dout_type);
    platform::ReorderMKLDNNHandler reorder_handler(
        x_vec_dims, dout->type(), dout_type, dev_ctx, onednn_engine, key);

    auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
        dx->format(), platform::to_void_cast(dout->data<T>()));
    auto reorder_dst_memory_p =
        reorder_handler.AcquireDstMemory(dx, dx->format(), ctx.GetPlace());
    auto reorder_p = reorder_handler.AcquireReorder(reorder_src_memory_p,
                                                    reorder_dst_memory_p);

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
    astream.wait();

    dx->Resize(x_dims);
    dx->set_layout(framework::DataLayout::kMKLDNN);
    dx->set_format(platform::GetMKLDNNFormat(*reorder_dst_memory_p));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(squeeze, MKLDNN, paddle::platform::CPUPlace,
                   ops::SqueezeMKLDNNKernel<float>,
                   ops::SqueezeMKLDNNKernel<paddle::platform::bfloat16>);

REGISTER_OP_KERNEL(squeeze_grad, MKLDNN, paddle::platform::CPUPlace,
                   ops::SqueezeGradMKLDNNKernel<float>,
                   ops::SqueezeGradMKLDNNKernel<paddle::platform::bfloat16>);

REGISTER_OP_KERNEL(squeeze2, MKLDNN, paddle::platform::CPUPlace,
                   ops::SqueezeMKLDNNKernel<float>,
                   ops::SqueezeMKLDNNKernel<paddle::platform::bfloat16>);

REGISTER_OP_KERNEL(squeeze2_grad, MKLDNN, paddle::platform::CPUPlace,
                   ops::SqueezeGradMKLDNNKernel<float>,
                   ops::SqueezeGradMKLDNNKernel<paddle::platform::bfloat16>);
