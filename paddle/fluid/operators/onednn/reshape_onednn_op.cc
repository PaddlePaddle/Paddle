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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/flatten_op.h"
#include "paddle/fluid/operators/squeeze_op.h"
#include "paddle/phi/backends/onednn/onednn_reuse.h"

namespace {
enum class ReshapeKernelOpName {
  reshape,
  squeeze,
  flatten,
};
}  // anonymous namespace

namespace paddle {
namespace operators {

static std::vector<int> extract_shape(
    const std::vector<const phi::DenseTensor*>& list_new_shape_tensor) {
  std::vector<int> vec_new_shape;
  vec_new_shape.reserve(list_new_shape_tensor.size());

  for (const auto& tensor : list_new_shape_tensor) {
    PADDLE_ENFORCE_EQ(
        tensor->dims(),
        common::make_ddim({1}),
        platform::errors::InvalidArgument(
            "If the element type of 'shape' in ReshapeOp is phi::DenseTensor, "
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
    const auto& dev_ctx = ctx.template device_context<phi::OneDNNContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* out = ctx.Output<phi::DenseTensor>("Out");

    framework::DDim x_dims, out_dims;
    InferInOutShape(ctx, x_dims, out_dims);

    auto x_vec_dims = common::vectorize(x_dims);

    auto x_type = phi::funcs ::ToOneDNNDataType(x->dtype());
    phi::funcs::ReorderOneDNNHandler reorder_handler(
        x_vec_dims, x->dtype(), x_type, onednn_engine);

    auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
        x->mem_desc(), phi::funcs::to_void_cast(x->data<T>()));
    out->Resize(x_dims);  // to match x numel, format is changed later
    // reorder is done into a plain tag to allow usage with blocked formats
    auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
        out, phi::funcs::GetPlainOneDNNFormat(x_dims.size()), ctx.GetPlace());
    auto reorder_p = reorder_handler.AcquireReorder(reorder_dst_memory_p,
                                                    reorder_src_memory_p);

    auto& astream = phi::OneDNNContext::tls().get_stream();
    reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);

    astream.wait();

    out->Resize(out_dims);
    auto reshape_dims = out_dims.size() != 0 ? common::vectorize(out_dims)
                                             : std::vector<int64_t>{1};
    out->set_mem_desc(reorder_dst_memory_p->get_desc().reshape(reshape_dims));
  }

  void InferInOutShape(const framework::ExecutionContext& ctx,
                       framework::DDim& x_dims,            // NOLINT
                       framework::DDim& out_dims) const {  // NOLINT
    switch (op_name) {
      case ReshapeKernelOpName::reshape:
        InferShapeReshapeOp(ctx, x_dims, out_dims);
        break;
      case ReshapeKernelOpName::squeeze:
        InferShapeSqueezeOp(ctx, x_dims, out_dims);
        break;
      case ReshapeKernelOpName::flatten:
      default:
        PADDLE_THROW(paddle::platform::errors::OutOfRange(
            "Reshape kernel doesn not support that operator name"));
    }
  }

  void InferShapeReshapeOp(const framework::ExecutionContext& ctx,
                           framework::DDim& x_dims,            // NOLINT
                           framework::DDim& out_dims) const {  // NOLINT
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* out = ctx.Output<phi::DenseTensor>("Out");
    x_dims = x->dims();
    out_dims = out->dims();
    ChangeReshapeOutDimsIfNeeded(ctx, x_dims, out_dims);
  }

  // in reshape1/2 ops  "ShapeTensor" has highest priority and "Shape" has
  // second highest priority
  void ChangeReshapeOutDimsIfNeeded(
      const framework::ExecutionContext& ctx,
      framework::DDim& x_dims,            // NOLINT
      framework::DDim& out_dims) const {  // NOLINT
    auto list_new_shape_tensor =
        ctx.MultiInput<phi::DenseTensor>("ShapeTensor");
    if (!list_new_shape_tensor.empty()) {
      auto new_shape = extract_shape(list_new_shape_tensor);
      out_dims = ValidateShape(new_shape, x_dims);
    } else if (ctx.HasInput("Shape")) {
      auto* shape_tensor = ctx.Input<phi::DenseTensor>("Shape");
      auto* shape_data = shape_tensor->data<int>();

      auto shape =
          std::vector<int>(shape_data, shape_data + shape_tensor->numel());
      out_dims = ValidateShape(shape, x_dims);
    }
  }

  void InferShapeSqueezeOp(const framework::ExecutionContext& ctx,
                           framework::DDim& x_dims,            // NOLINT
                           framework::DDim& out_dims) const {  // NOLINT
    auto* x = ctx.Input<phi::DenseTensor>("X");
    x_dims = x->dims();
    const auto& axes = ctx.Attr<std::vector<int>>("axes");
    out_dims = GetOutputShape(axes, x_dims, true);
  }

  void InferShapeFlattenOp(const framework::ExecutionContext& ctx,
                           framework::DDim& x_dims,            // NOLINT
                           framework::DDim& out_dims) const {  // NOLINT
    auto x = ctx.Input<phi::DenseTensor>("X");
    x_dims = x->dims();
    auto axes = ctx.Attr<int>("axis");
    out_dims = common::make_ddim(
        Flatten2Kernel<phi::CPUContext, float>::GetOutputShape(axes, x_dims));
  }

 protected:
  static framework::DDim ValidateShape(const std::vector<int>& shape,
                                       const framework::DDim& in_dims) {
    const int64_t in_size = common::product(in_dims);
    auto in_dims_vec = common::vectorize(in_dims);
    bool all_positive = std::all_of(in_dims_vec.cbegin(),
                                    in_dims_vec.cend(),
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
            unk_dim_idx,
            -1,
            platform::errors::InvalidArgument(
                "Only one dimension value of 'shape' in ReshapeOp can "
                "be -1. But received shape = [%s], shape[%d] is also -1.",
                common::make_ddim(shape),
                i));
        unk_dim_idx = static_cast<int>(i);
      } else if (shape[i] == copy_dim_val) {
        PADDLE_ENFORCE_LT(
            static_cast<int>(i),
            in_dims.size(),
            platform::errors::InvalidArgument(
                "The index of 0 in `shape` must be less than "
                "the input tensor X's dimensions. "
                "But received shape = [%s], shape[%d] = 0, X's shape = [%s], "
                "X's dimensions = %d.",
                common::make_ddim(shape),
                i,
                in_dims,
                in_dims.size()));
      } else {
        PADDLE_ENFORCE_GT(
            shape[i],
            0,
            platform::errors::InvalidArgument(
                "Each dimension value of 'shape' in ReshapeOp must not "
                "be negative except one unknown dimension. "
                "But received  shape = [%s], shape[%d] = %d.",
                common::make_ddim(shape),
                i,
                shape[i]));
      }

      capacity *= (shape[i] ? shape[i] : in_dims[i]);  // NOLINT
      output_shape[i] =
          (shape[i] ? static_cast<int64_t>(shape[i]) : in_dims[i]);  // NOLINT
    }

    if (unk_dim_idx != -1) {
      if (all_positive) {
        // in_size < 0 and is un-determinate in compile time, skip the check,
        // for example, in_dims = [-1, 8, 1, 1], shape = [-1, 3, 8],
        // capacity = -24, in_size = -8, output_shape[0] = 0
        // the following check will fail.
        output_shape[unk_dim_idx] = -in_size / capacity;
        PADDLE_ENFORCE_EQ(
            output_shape[unk_dim_idx] * capacity,
            -in_size,
            platform::errors::InvalidArgument(
                "The 'shape' attribute in ReshapeOp is invalid. "
                "The input tensor X'size must be divisible by known "
                "capacity of 'shape'. "
                "But received X's shape = [%s], X's size = %d, "
                "'shape' is [%s], known capacity of 'shape' is %d.",
                in_dims,
                in_size,
                common::make_ddim(shape),
                capacity));
      } else {
        output_shape[unk_dim_idx] = -1;
      }
    } else {
      if (all_positive) {
        PADDLE_ENFORCE_EQ(
            capacity,
            in_size,
            platform::errors::InvalidArgument(
                "The 'shape' in ReshapeOp is invalid. "
                "The input tensor X'size must be equal to the capacity of "
                "'shape'. "
                "But received X's shape = [%s], X's size = %d, 'shape' is "
                "[%s], the capacity of 'shape' is %d.",
                in_dims,
                in_size,
                common::make_ddim(shape),
                capacity));
      }
    }
    return common::make_ddim(output_shape);
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
    const auto& dev_ctx = ctx.template device_context<phi::OneDNNContext>();
    const auto& onednn_engine = dev_ctx.GetEngine();

    auto* dout = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));

    framework::DDim dx_dims;
    InferOutputShapeInGrad(ctx, dx_dims);

    auto dout_vec_dims = dout->dims().size() != 0
                             ? common::vectorize(dout->dims())
                             : std::vector<int64_t>{1};

    auto dout_type = phi::funcs::ToOneDNNDataType(dout->dtype());
    phi::funcs::ReorderOneDNNHandler reorder_handler(
        dout_vec_dims, dout->dtype(), dout_type, onednn_engine);

    auto reorder_src_memory_p = reorder_handler.AcquireSrcMemory(
        dout->mem_desc(), phi::funcs::to_void_cast(dout->data<T>()));
    auto reorder_dst_memory_p = reorder_handler.AcquireDstMemory(
        dx,
        phi::funcs::GetPlainOneDNNFormat(dout_vec_dims.size()),
        ctx.GetPlace());
    auto reorder_p = reorder_handler.AcquireReorder(reorder_dst_memory_p,
                                                    reorder_src_memory_p);

    auto& astream = phi::OneDNNContext::tls().get_stream();
    reorder_p->execute(astream, *reorder_src_memory_p, *reorder_dst_memory_p);
    astream.wait();

    dx->Resize(dx_dims);
    const auto reshape_dims = dx_dims.size() != 0 ? common::vectorize(dx_dims)
                                                  : std::vector<int64_t>{1};
    reorder_dst_memory_p->get_desc().reshape(reshape_dims);
  }

  void InferOutputShapeInGrad(const framework::ExecutionContext& ctx,
                              framework::DDim& x_dims) const {  // NOLINT
    switch (op_name) {
      case ReshapeKernelOpName::reshape:
        InferShapeReshapeSqueezeGradOp(ctx, x_dims);
        break;
      case ReshapeKernelOpName::squeeze:
        InferShapeReshapeSqueezeGradOp(ctx, x_dims);
        break;
      case ReshapeKernelOpName::flatten:
        InferShapeFlattenGradOp(ctx, x_dims);
        break;
      default:
        PADDLE_THROW(paddle::platform::errors::OutOfRange(
            "Reshape grad kernel doesn not support that operator name"));
    }
  }

  void InferShapeReshapeSqueezeGradOp(
      const framework::ExecutionContext& ctx,
      framework::DDim& dx_dims) const {  // NOLINT
    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    dx_dims = dx->dims();
  }

  void InferShapeFlattenGradOp(const framework::ExecutionContext& ctx,
                               framework::DDim& dx_dims) const {  // NOLINT
    dx_dims = ctx.Input<phi::DenseTensor>("X")->dims();
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL(
    squeeze,
    MKLDNN,
    phi::CPUPlace,
    ops::ReshapeMKLDNNKernel<float, ReshapeKernelOpName::squeeze>,
    ops::ReshapeMKLDNNKernel<paddle::platform::bfloat16,
                             ReshapeKernelOpName::squeeze>);

REGISTER_OP_KERNEL(
    squeeze_grad,
    MKLDNN,
    phi::CPUPlace,
    ops::ReshapeGradMKLDNNKernel<float, ReshapeKernelOpName::squeeze>,
    ops::ReshapeGradMKLDNNKernel<paddle::platform::bfloat16,
                                 ReshapeKernelOpName::squeeze>);

REGISTER_OP_KERNEL(
    reshape,
    MKLDNN,
    phi::CPUPlace,
    ops::ReshapeMKLDNNKernel<float, ReshapeKernelOpName::reshape>,
    ops::ReshapeMKLDNNKernel<paddle::platform::bfloat16,
                             ReshapeKernelOpName::reshape>);

REGISTER_OP_KERNEL(
    reshape_grad,
    MKLDNN,
    phi::CPUPlace,
    ops::ReshapeGradMKLDNNKernel<float, ReshapeKernelOpName::reshape>,
    ops::ReshapeGradMKLDNNKernel<paddle::platform::bfloat16,
                                 ReshapeKernelOpName::reshape>);

REGISTER_OP_KERNEL(
    flatten,
    MKLDNN,
    phi::CPUPlace,
    ops::ReshapeMKLDNNKernel<float, ReshapeKernelOpName::flatten>,
    ops::ReshapeMKLDNNKernel<paddle::platform::bfloat16,
                             ReshapeKernelOpName::flatten>);

REGISTER_OP_KERNEL(
    flatten_grad,
    MKLDNN,
    phi::CPUPlace,
    ops::ReshapeGradMKLDNNKernel<float, ReshapeKernelOpName::flatten>,
    ops::ReshapeGradMKLDNNKernel<paddle::platform::bfloat16,
                                 ReshapeKernelOpName::flatten>);
