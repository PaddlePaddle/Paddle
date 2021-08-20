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

using paddle::framework::LoDTensor;
using platform::to_void_cast;
using platform::GetMKLDNNFormat;

template <typename T>
class ReshapeMKLDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    RunKernel(ctx);
  }

 private:
  void RunKernel(const framework::ExecutionContext& ctx) const {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto* x = ctx.Input<LoDTensor>("X");
    auto* xshape = ctx.Output<LoDTensor>("XShape");
    auto* out = ctx.Output<LoDTensor>("Out");

    framework::DDim x_dims;
    // if reshape or squeeze
    if (ctx.Type().find("2") == std::string::npos) {
      x_dims = x->dims();
    } else {
      auto xshape_dims = xshape->dims();
      x_dims = framework::slice_ddim(xshape_dims, 1, xshape_dims.size());
    }

    auto x_vec_dims = framework::vectorize(x_dims);

    framework::DDim out_dims;
    if (ctx.Type() == "squeeze") {
      auto& axes = ctx.Attr<std::vector<int>>("axes");
      out_dims = GetOutputShape(axes, x_dims, true);
    } else {
      out_dims = out->dims();
    }

    if (ctx.Type().find("reshape") != std::string::npos) {
      if (ctx.HasInput("Shape")) {
        auto* shape_tensor = ctx.Input<framework::LoDTensor>("Shape");
        auto* shape_data = shape_tensor->data<int>();

        auto shape =
            std::vector<int>(shape_data, shape_data + shape_tensor->numel());
        out_dims = ValidateShape(shape, x_dims);
      }
    }

    mkldnn::memory::data_type x_type = framework::ToMKLDNNDataType(x->type());
    std::string key =
        platform::CreateKey(dev_ctx, x_vec_dims, x->format(), x_type);
    platform::ReorderMKLDNNHandler reorder_handler(
        x_vec_dims, x->type(), x_type, dev_ctx, onednn_engine, key);

    auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
        x->format(), platform::to_void_cast(x->data<T>()));
    out->Resize(x_dims);  // to match x numel, format is changed later
    // reorder is done into a plain tag to allow usage with blocked formats
    auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
        out, getPlainFormatTag(x), ctx.GetPlace());
    auto reorder_p = reorder_handler.AcquireReorder(reorder_src_memory_p,
                                                    reorder_dst_memory_p);

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);

    astream.wait();

    out->Resize(out_dims);
    out->set_layout(framework::DataLayout::kMKLDNN);
    out->set_format(GetMKLDNNFormat(reorder_dst_memory_p->get_desc().reshape(
        framework::vectorize(out_dims))));
  }

 protected:
  static mkldnn::memory::format_tag getPlainFormatTag(const Tensor* tensor) {
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
      default:
        return mkldnn::memory::format_tag::abcdef;
    }
  }

  static framework::DDim ValidateShape(const std::vector<int>& shape,
                                       const framework::DDim& in_dims) {
    const int64_t in_size = framework::product(in_dims);
    auto in_dims_vec = framework::vectorize(in_dims);
    bool all_positive = std::all_of(in_dims_vec.cbegin(), in_dims_vec.cend(),
                                    [](int64_t i) { return i > 0; });
    // only one dimension can be set to -1, whose size will be automatically
    // infered
    const int64_t unk_dim_val = -1;
    const int64_t copy_dim_val = 0;

    std::vector<int64_t> output_shape(shape.size(), 0);
    int64_t capacity = 1;
    int unk_dim_idx = -1;
    for (size_t i = 0; i < shape.size(); ++i) {
      if (shape[i] == unk_dim_val) {
        PADDLE_ENFORCE_EQ(
            unk_dim_idx, -1,
            platform::errors::InvalidArgument(
                "Only one dimension value of 'shape' in ReshapeOp can "
                "be -1. But received shape = [%s], shape[%d] is also -1.",
                framework::make_ddim(shape), i));
        unk_dim_idx = i;
      } else if (shape[i] == copy_dim_val) {
        PADDLE_ENFORCE_LT(
            static_cast<int>(i), in_dims.size(),
            platform::errors::InvalidArgument(
                "The index of 0 in `shape` must be less than "
                "the input tensor X's dimensions. "
                "But received shape = [%s], shape[%d] = 0, X's shape = [%s], "
                "X's dimensions = %d.",
                framework::make_ddim(shape), i, in_dims, in_dims.size()));
      } else {
        PADDLE_ENFORCE_GT(
            shape[i], 0,
            platform::errors::InvalidArgument(
                "Each dimension value of 'shape' in ReshapeOp must not "
                "be negative except one unknown dimension. "
                "But received  shape = [%s], shape[%d] = %d.",
                framework::make_ddim(shape), i, shape[i]));
      }

      capacity *= (shape[i] ? shape[i] : in_dims[i]);
      output_shape[i] =
          (shape[i] ? static_cast<int64_t>(shape[i]) : in_dims[i]);
    }

    if (unk_dim_idx != -1) {
      if (all_positive) {
        // in_size < 0 and is un-determinate in compile time, skip the check,
        // for example, in_dims = [-1, 8, 1, 1], shape = [-1, 3, 8],
        // capacity = -24, in_size = -8, output_shape[0] = 0
        // the following check will fail.
        output_shape[unk_dim_idx] = -in_size / capacity;
        PADDLE_ENFORCE_EQ(
            output_shape[unk_dim_idx] * capacity, -in_size,
            platform::errors::InvalidArgument(
                "The 'shape' attribute in ReshapeOp is invalid. "
                "The input tensor X'size must be divisible by known "
                "capacity of 'shape'. "
                "But received X's shape = [%s], X's size = %d, "
                "'shape' is [%s], known capacity of 'shape' is %d.",
                in_dims, in_size, framework::make_ddim(shape), capacity));
      } else {
        output_shape[unk_dim_idx] = -1;
      }
    } else {
      if (all_positive) {
        PADDLE_ENFORCE_EQ(
            capacity, in_size,
            platform::errors::InvalidArgument(
                "The 'shape' in ReshapeOp is invalid. "
                "The input tensor X'size must be equal to the capacity of "
                "'shape'. "
                "But received X's shape = [%s], X's size = %d, 'shape' is "
                "[%s], the capacity of 'shape' is %d.",
                in_dims, in_size, framework::make_ddim(shape), capacity));
      }
    }
    return framework::make_ddim(output_shape);
  }
};

template <typename T>
class ReshapeGradMKLDNNKernel : public ReshapeMKLDNNKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    RunKernel(ctx);
  }

 private:
  void RunKernel(const framework::ExecutionContext& ctx) const {
    const auto& dev_ctx =
        ctx.template device_context<platform::MKLDNNDeviceContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto* dout = ctx.Input<LoDTensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<LoDTensor>(framework::GradVarName("X"));

    framework::DDim x_dims;
    // if reshape or squeeze
    if (ctx.Type().find("2") == std::string::npos) {
      x_dims = dx->dims();
    } else {
      auto xshape_dims = ctx.Input<framework::LoDTensor>("XShape")->dims();
      x_dims = framework::slice_ddim(xshape_dims, 1, xshape_dims.size());
    }
    auto dout_vec_dims = framework::vectorize(dout->dims());

    mkldnn::memory::data_type dout_type =
        framework::ToMKLDNNDataType(dout->type());
    std::string key =
        platform::CreateKey(dev_ctx, dout_vec_dims, this->getPlainFormatTag(dx),
                            dx->format(), dout_type);
    platform::ReorderMKLDNNHandler reorder_handler(
        dout_vec_dims, dout->type(), dout_type, dev_ctx, onednn_engine, key);

    auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
        dout->format(), platform::to_void_cast(dout->data<T>()));
    auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
        dx, this->getPlainFormatTag(dout), ctx.GetPlace());
    auto reorder_p = reorder_handler.AcquireReorder(reorder_src_memory_p,
                                                    reorder_dst_memory_p);

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
    astream.wait();

    dx->Resize(x_dims);
    dx->set_layout(framework::DataLayout::kMKLDNN);
    dx->set_format(GetMKLDNNFormat(reorder_dst_memory_p->get_desc().reshape(
        framework::vectorize(x_dims))));
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(squeeze, MKLDNN, paddle::platform::CPUPlace,
                   ops::ReshapeMKLDNNKernel<float>,
                   ops::ReshapeMKLDNNKernel<paddle::platform::bfloat16>);

REGISTER_OP_KERNEL(squeeze_grad, MKLDNN, paddle::platform::CPUPlace,
                   ops::ReshapeGradMKLDNNKernel<float>,
                   ops::ReshapeGradMKLDNNKernel<paddle::platform::bfloat16>);

REGISTER_OP_KERNEL(squeeze2, MKLDNN, paddle::platform::CPUPlace,
                   ops::ReshapeMKLDNNKernel<float>,
                   ops::ReshapeMKLDNNKernel<paddle::platform::bfloat16>);

REGISTER_OP_KERNEL(squeeze2_grad, MKLDNN, paddle::platform::CPUPlace,
                   ops::ReshapeGradMKLDNNKernel<float>,
                   ops::ReshapeGradMKLDNNKernel<paddle::platform::bfloat16>);

REGISTER_OP_KERNEL(reshape, MKLDNN, paddle::platform::CPUPlace,
                   ops::ReshapeMKLDNNKernel<float>,
                   ops::ReshapeMKLDNNKernel<paddle::platform::bfloat16>);

REGISTER_OP_KERNEL(reshape_grad, MKLDNN, paddle::platform::CPUPlace,
                   ops::ReshapeGradMKLDNNKernel<float>,
                   ops::ReshapeGradMKLDNNKernel<paddle::platform::bfloat16>);

REGISTER_OP_KERNEL(reshape2, MKLDNN, paddle::platform::CPUPlace,
                   ops::ReshapeMKLDNNKernel<float>,
                   ops::ReshapeMKLDNNKernel<paddle::platform::bfloat16>);

REGISTER_OP_KERNEL(reshape2_grad, MKLDNN, paddle::platform::CPUPlace,
                   ops::ReshapeGradMKLDNNKernel<float>,
                   ops::ReshapeGradMKLDNNKernel<paddle::platform::bfloat16>);
