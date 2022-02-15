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

#include "paddle/fluid/operators/flatten_op.h"
#include "paddle/fluid/operators/squeeze_op.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"

namespace {
enum class ReshapeKernelOpName {
  reshape,
  reshape2,
  squeeze,
  squeeze2,
  flatten,
  flatten2,
};
}  // anonymous namespace

namespace paddle {
namespace operators {

using paddle::framework::LoDTensor;
using platform::to_void_cast;
using platform::GetMKLDNNFormat;

static std::vector<int> extract_shape(
    const std::vector<const Tensor*>& list_new_shape_tensor) {
  std::vector<int> vec_new_shape;
  vec_new_shape.reserve(list_new_shape_tensor.size());

  for (const auto& tensor : list_new_shape_tensor) {
    PADDLE_ENFORCE_EQ(
        tensor->dims(), framework::make_ddim({1}),
        platform::errors::InvalidArgument(
            "If the element type of 'shape' in ReshapeOp is Tensor, "
            "the element's shape must be [1]. But received the element's shape "
            "is [%s]",
            tensor->dims()));
    vec_new_shape.emplace_back(*tensor->data<int32_t>());
  }

  return vec_new_shape;
}

template <typename T, ReshapeKernelOpName op_name>
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
    auto* out = ctx.Output<LoDTensor>("Out");

    framework::DDim x_dims, out_dims;
    InferInOutShape(ctx, x_dims, out_dims);

    auto x_vec_dims = framework::vectorize(x_dims);

    dnnl::memory::data_type x_type =
        framework::ToMKLDNNDataType(framework::TransToProtoVarType(x->dtype()));
    platform::ReorderMKLDNNHandler reorder_handler(
        x_vec_dims, framework::TransToProtoVarType(x->dtype()), x_type,
        onednn_engine);

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

  void InferInOutShape(const framework::ExecutionContext& ctx,
                       framework::DDim& x_dims,            // NOLINT
                       framework::DDim& out_dims) const {  // NOLINT
    switch (op_name) {
      case ReshapeKernelOpName::reshape:
        InferShapeReshapeOp(ctx, x_dims, out_dims);
        break;
      case ReshapeKernelOpName::reshape2:
        InferShapeReshape2Op(ctx, x_dims, out_dims);
        break;
      case ReshapeKernelOpName::squeeze:
        InferShapeSqueezeOp(ctx, x_dims, out_dims);
        break;
      case ReshapeKernelOpName::squeeze2:
        InferShapeSqueeze2Op(ctx, x_dims, out_dims);
        break;
      case ReshapeKernelOpName::flatten:
        InferShapeFlattenOp(ctx, x_dims, out_dims);
        break;
      case ReshapeKernelOpName::flatten2:
        InferShapeFlattenOp(ctx, x_dims, out_dims);
        break;
      default:
        PADDLE_THROW(paddle::platform::errors::OutOfRange(
            "Reshape kernel doesn not support that operator name"));
    }
  }

  void InferShapeReshapeOp(const framework::ExecutionContext& ctx,
                           framework::DDim& x_dims,            // NOLINT
                           framework::DDim& out_dims) const {  // NOLINT
    auto* x = ctx.Input<LoDTensor>("X");
    auto* out = ctx.Output<LoDTensor>("Out");
    x_dims = x->dims();
    out_dims = out->dims();
    ChangeReshapeOutDimsIfNeeded(ctx, x_dims, out_dims);
  }

  void InferShapeReshape2Op(const framework::ExecutionContext& ctx,
                            framework::DDim& x_dims,            // NOLINT
                            framework::DDim& out_dims) const {  // NOLINT
    auto* out = ctx.Output<LoDTensor>("Out");
    auto* xshape = ctx.Output<LoDTensor>("XShape");
    auto xshape_dims = xshape->dims();
    x_dims = framework::slice_ddim(xshape_dims, 1, xshape_dims.size());
    out_dims = out->dims();
    ChangeReshapeOutDimsIfNeeded(ctx, x_dims, out_dims);
  }

  // in reshape1/2 ops  "ShapeTensor" has highest priority and "Shape" has
  // second highest priority
  void ChangeReshapeOutDimsIfNeeded(
      const framework::ExecutionContext& ctx,
      framework::DDim& x_dims,            // NOLINT
      framework::DDim& out_dims) const {  // NOLINT
    auto list_new_shape_tensor = ctx.MultiInput<Tensor>("ShapeTensor");
    if (list_new_shape_tensor.size() > 0) {
      auto new_shape = extract_shape(list_new_shape_tensor);
      out_dims = ValidateShape(new_shape, x_dims);
    } else if (ctx.HasInput("Shape")) {
      auto* shape_tensor = ctx.Input<framework::LoDTensor>("Shape");
      auto* shape_data = shape_tensor->data<int>();

      auto shape =
          std::vector<int>(shape_data, shape_data + shape_tensor->numel());
      out_dims = ValidateShape(shape, x_dims);
    }
  }

  void InferShapeSqueezeOp(const framework::ExecutionContext& ctx,
                           framework::DDim& x_dims,            // NOLINT
                           framework::DDim& out_dims) const {  // NOLINT
    auto* x = ctx.Input<LoDTensor>("X");
    x_dims = x->dims();
    const auto& axes = ctx.Attr<std::vector<int>>("axes");
    out_dims = GetOutputShape(axes, x_dims, true);
  }

  void InferShapeSqueeze2Op(const framework::ExecutionContext& ctx,
                            framework::DDim& x_dims,            // NOLINT
                            framework::DDim& out_dims) const {  // NOLINT
    auto* out = ctx.Output<LoDTensor>("Out");
    auto* xshape = ctx.Output<LoDTensor>("XShape");
    auto xshape_dims = xshape->dims();
    x_dims = framework::slice_ddim(xshape_dims, 1, xshape_dims.size());
    out_dims = out->dims();
  }

  void InferShapeFlattenOp(const framework::ExecutionContext& ctx,
                           framework::DDim& x_dims,            // NOLINT
                           framework::DDim& out_dims) const {  // NOLINT
    auto x = ctx.Input<LoDTensor>("X");
    x_dims = x->dims();
    auto axes = ctx.Attr<int>("axis");
    out_dims = framework::make_ddim(
        FlattenKernel<platform::CPUDeviceContext, float>::GetOutputShape(
            axes, x_dims));
  }

 protected:
  static dnnl::memory::format_tag getPlainFormatTag(const Tensor* tensor) {
    auto tensor_dims_size = tensor->dims().size();
    PADDLE_ENFORCE_EQ(
        tensor_dims_size <= 6 && tensor_dims_size >= 1, true,
        platform::errors::InvalidArgument(
            "Dims for squeeze_grad oneDNN op must be in range <1, 6>"));

    switch (tensor_dims_size) {
      case 1:
        return dnnl::memory::format_tag::a;
      case 2:
        return dnnl::memory::format_tag::ab;
      case 3:
        return dnnl::memory::format_tag::abc;
      case 4:
        return dnnl::memory::format_tag::abcd;
      case 5:
        return dnnl::memory::format_tag::abcde;
      default:
        return dnnl::memory::format_tag::abcdef;
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

template <typename T, ReshapeKernelOpName op_name>
class ReshapeGradMKLDNNKernel : public ReshapeMKLDNNKernel<T, op_name> {
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

    framework::DDim dx_dims;
    InferOutputShapeInGrad(ctx, dx_dims);

    auto dout_vec_dims = framework::vectorize(dout->dims());

    dnnl::memory::data_type dout_type = framework::ToMKLDNNDataType(
        framework::TransToProtoVarType(dout->dtype()));
    platform::ReorderMKLDNNHandler reorder_handler(
        dout_vec_dims, framework::TransToProtoVarType(dout->dtype()), dout_type,
        onednn_engine);

    auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
        dout->format(), platform::to_void_cast(dout->data<T>()));
    auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
        dx, this->getPlainFormatTag(dout), ctx.GetPlace());
    auto reorder_p = reorder_handler.AcquireReorder(reorder_src_memory_p,
                                                    reorder_dst_memory_p);

    auto& astream = platform::MKLDNNDeviceContext::tls().get_stream();
    reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
    astream.wait();

    dx->Resize(dx_dims);
    dx->set_layout(framework::DataLayout::kMKLDNN);
    dx->set_format(GetMKLDNNFormat(reorder_dst_memory_p->get_desc().reshape(
        framework::vectorize(dx_dims))));
  }

  void InferOutputShapeInGrad(const framework::ExecutionContext& ctx,
                              framework::DDim& x_dims) const {  // NOLINT
    switch (op_name) {
      case ReshapeKernelOpName::reshape:
        InferShapeReshapeSqueezeGradOp(ctx, x_dims);
        break;
      case ReshapeKernelOpName::reshape2:
        InferShapeReshape2Squeeze2Flatten2GradOp(ctx, x_dims);
        break;
      case ReshapeKernelOpName::squeeze:
        InferShapeReshapeSqueezeGradOp(ctx, x_dims);
        break;
      case ReshapeKernelOpName::squeeze2:
        InferShapeReshape2Squeeze2Flatten2GradOp(ctx, x_dims);
        break;
      case ReshapeKernelOpName::flatten:
        InferShapeFlattenGradOp(ctx, x_dims);
        break;
      case ReshapeKernelOpName::flatten2:
        InferShapeReshape2Squeeze2Flatten2GradOp(ctx, x_dims);
        break;
      default:
        PADDLE_THROW(paddle::platform::errors::OutOfRange(
            "Reshape grad kernel doesn not support that operator name"));
    }
  }

  void InferShapeReshapeSqueezeGradOp(
      const framework::ExecutionContext& ctx,
      framework::DDim& dx_dims) const {  // NOLINT
    auto* dx = ctx.Output<LoDTensor>(framework::GradVarName("X"));
    dx_dims = dx->dims();
  }

  void InferShapeReshape2Squeeze2Flatten2GradOp(
      const framework::ExecutionContext& ctx,
      framework::DDim& dx_dims) const {  // NOLINT
    auto xshape_dims = ctx.Input<framework::LoDTensor>("XShape")->dims();
    dx_dims = framework::slice_ddim(xshape_dims, 1, xshape_dims.size());
  }

  void InferShapeFlattenGradOp(const framework::ExecutionContext& ctx,
                               framework::DDim& dx_dims) const {  // NOLINT
    dx_dims = ctx.Input<LoDTensor>("X")->dims();
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(
    squeeze, MKLDNN, paddle::platform::CPUPlace,
    ops::ReshapeMKLDNNKernel<float, ReshapeKernelOpName::squeeze>,
    ops::ReshapeMKLDNNKernel<paddle::platform::bfloat16,
                             ReshapeKernelOpName::squeeze>);

REGISTER_OP_KERNEL(
    squeeze_grad, MKLDNN, paddle::platform::CPUPlace,
    ops::ReshapeGradMKLDNNKernel<float, ReshapeKernelOpName::squeeze>,
    ops::ReshapeGradMKLDNNKernel<paddle::platform::bfloat16,
                                 ReshapeKernelOpName::squeeze>);

REGISTER_OP_KERNEL(
    squeeze2, MKLDNN, paddle::platform::CPUPlace,
    ops::ReshapeMKLDNNKernel<float, ReshapeKernelOpName::squeeze2>,
    ops::ReshapeMKLDNNKernel<paddle::platform::bfloat16,
                             ReshapeKernelOpName::squeeze2>);

REGISTER_OP_KERNEL(
    squeeze2_grad, MKLDNN, paddle::platform::CPUPlace,
    ops::ReshapeGradMKLDNNKernel<float, ReshapeKernelOpName::squeeze2>,
    ops::ReshapeGradMKLDNNKernel<paddle::platform::bfloat16,
                                 ReshapeKernelOpName::squeeze2>);

REGISTER_OP_KERNEL(
    reshape, MKLDNN, paddle::platform::CPUPlace,
    ops::ReshapeMKLDNNKernel<float, ReshapeKernelOpName::reshape>,
    ops::ReshapeMKLDNNKernel<paddle::platform::bfloat16,
                             ReshapeKernelOpName::reshape>);

REGISTER_OP_KERNEL(
    reshape_grad, MKLDNN, paddle::platform::CPUPlace,
    ops::ReshapeGradMKLDNNKernel<float, ReshapeKernelOpName::reshape>,
    ops::ReshapeGradMKLDNNKernel<paddle::platform::bfloat16,
                                 ReshapeKernelOpName::reshape>);

REGISTER_OP_KERNEL(
    reshape2, MKLDNN, paddle::platform::CPUPlace,
    ops::ReshapeMKLDNNKernel<float, ReshapeKernelOpName::reshape2>,
    ops::ReshapeMKLDNNKernel<paddle::platform::bfloat16,
                             ReshapeKernelOpName::reshape2>);

REGISTER_OP_KERNEL(
    reshape2_grad, MKLDNN, paddle::platform::CPUPlace,
    ops::ReshapeGradMKLDNNKernel<float, ReshapeKernelOpName::reshape2>,
    ops::ReshapeGradMKLDNNKernel<paddle::platform::bfloat16,
                                 ReshapeKernelOpName::reshape2>);

REGISTER_OP_KERNEL(
    flatten, MKLDNN, paddle::platform::CPUPlace,
    ops::ReshapeMKLDNNKernel<float, ReshapeKernelOpName::flatten>,
    ops::ReshapeMKLDNNKernel<paddle::platform::bfloat16,
                             ReshapeKernelOpName::flatten>);

REGISTER_OP_KERNEL(
    flatten_grad, MKLDNN, paddle::platform::CPUPlace,
    ops::ReshapeGradMKLDNNKernel<float, ReshapeKernelOpName::flatten>,
    ops::ReshapeGradMKLDNNKernel<paddle::platform::bfloat16,
                                 ReshapeKernelOpName::flatten>);

REGISTER_OP_KERNEL(
    flatten2, MKLDNN, paddle::platform::CPUPlace,
    ops::ReshapeMKLDNNKernel<float, ReshapeKernelOpName::flatten2>,
    ops::ReshapeMKLDNNKernel<paddle::platform::bfloat16,
                             ReshapeKernelOpName::flatten2>);

REGISTER_OP_KERNEL(
    flatten2_grad, MKLDNN, paddle::platform::CPUPlace,
    ops::ReshapeGradMKLDNNKernel<float, ReshapeKernelOpName::flatten2>,
    ops::ReshapeGradMKLDNNKernel<paddle::platform::bfloat16,
                                 ReshapeKernelOpName::flatten2>);
