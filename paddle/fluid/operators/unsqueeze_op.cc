/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/unsqueeze_op.h"
#include <string>
#include <vector>

namespace paddle {
namespace operators {

using framework::OpKernelType;
using framework::Tensor;

class UnsqueezeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of UnsqueezeOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of UnsqueezeOp should not be null.");

    const auto& axes = ctx->Attrs().Get<std::vector<int>>("axes");
    PADDLE_ENFORCE(!axes.empty(),
                   "The unsqueeze axes information must be set by Attr(axes).");

    const auto& x_dims = ctx->GetInputDim("X");
    // Validity Check: input tensor dims (<6).
    PADDLE_ENFORCE(x_dims.size() < 6,
                   "Invalid dimensions, dynamic dimensions should within "
                   "[0, 5] dimensions (Eigen limit).");
    // Validity Check: the range of unsqueeze aixs.
    // TODO(chenweihang): Don't consider negative axis?.
    for (unsigned int idx = 0; idx < axes.size(); ++idx) {
      PADDLE_ENFORCE(axes[idx] < 6,
                     "Invalid dimensions, input axis should within "
                     "[0, 5] dimensions (Eigen limit).");
    }

    auto out_dims = GetOutputShape(axes, x_dims);
    ctx->SetOutputDim("Out", out_dims);
  }

  static framework::DDim GetOutputShape(const std::vector<int> unsqz_dims,
                                        const framework::DDim& in_dims) {
    /*
     * STL version
     * Test Error! don't know why?.
    std::vector<int64_t> output_shape;

    // Contruct base output shape
    for(int idx = 0; idx < in_dims.size(); ++idx) {
      output_shape.emplace_back(in_dims[idx]);
    }
    // Validity Check: output dimensions limit.
    PADDLE_ENFORCE(unsqz_dims.size() + output_shape.size() < 6,
                   "The Attr(axes) size is too large. The output shape should "
                   "be less than 6 (Eigne limit).");
    // Insert the unsqueeze axis in turn.
    auto it = output_shape.begin();
    for (int axis : unsqz_dims) {
      int cur = axis < 0 ? (axis + output_shape.size() + 1)
                         : axis;
      // Vaildity Check: the axis bound
      PADDLE_ENFORCE(cur >= 0 && cur <= static_cast<int>(output_shape.size()),
                     "The unsqueeze dims must be within range of current
    rank.");
      output_shape.emplace(it + axis, 1);
    }
    */

    unsigned int unsqz_mask = 0;
    unsigned int front = 0, back = 0;
    int output_dims_size = in_dims.size();

    // Simulate insert by bit calc.
    for (int axis : unsqz_dims) {
      int cur = axis < 0 ? axis + output_dims_size + 1 : axis;
      // Vaildity Check: the axis bound
      PADDLE_ENFORCE(
          cur >= 0 && cur <= output_dims_size,
          "The unsqueeze dims must be within range of current rank.");
      // Save the front part.
      front = unsqz_mask & ((1 << axis) - 1);
      // Move the back part.
      back = unsqz_mask & ~((1 << axis) - 1);
      back <<= 1;
      // Merge two part.
      back |= (1 << axis);
      unsqz_mask = front | back;
      // Add the output size.
      output_dims_size++;
      // Validity Check: rank range.
      PADDLE_ENFORCE(output_dims_size < 6,
                     "The output tensor's rank should be less than 6.");
    }

    // Make output shape
    std::vector<int64_t> output_shape(output_dims_size, 0);
    for (int in_idx = 0, out_idx = 0; out_idx < output_dims_size; ++out_idx) {
      if ((unsqz_mask & (1 << out_idx)) == 0) {
        output_shape[out_idx] = in_dims[in_idx++];
      } else {
        output_shape[out_idx] = 1;
      }
    }

    return framework::make_ddim(output_shape);
  }
};

class UnsqueezeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor). The input tensor of unsqueeze operator.");
    AddOutput("Out", "(Tensor). The output tensor of unsqueeze operator.");
    AddAttr<std::vector<int>>("axes",
                              "(std::vector<int>). List of positive integers,"
                              " indicate the dimensions to be inserted");
    AddAttr<bool>(
        "inplace",
        "(default: false) Unsqueeze the source tensor's shape without "
        "memory copy. When Attr(inplace) is set true, the output "
        "tensor shares memory with Input(X), otherwise, a new output "
        "tensor is created, and its data are copied from Input(x).")
        .SetDefault(false);
    AddComment(R"DOC(
    Unsqueeze Operator.
    
    Insert single-dimensional entries to the shape of a tensor. 
    Takes one required argument axes, a list of dimensions that will be inserted. 
    Dimension indices in axes are as seen in the output tensor. 

    For example: 
      Given a tensor such that tensor with shape [3, 4, 5], 
      then Unsqueeze(tensor, axes=[0, 4]) has shape [1, 3, 4, 5, 1]
    )DOC");
  }
};

class UnsqueezeGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of UnsqueezeGradOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Output(Out@GRAD) of UnsqueezeGradOp should not be null.");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::LoDTensor>("X")->type()),
        ctx.device_context());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(unsqueeze, ops::UnsqueezeOp, ops::UnsqueezeOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(unsqueeze_grad, ops::UnsqueezeGradOp);
REGISTER_OP_CPU_KERNEL(
    unsqueeze, ops::UnsqueezeKernel<paddle::platform::CPUDeviceContext, float>,
    ops::UnsqueezeKernel<paddle::platform::CPUDeviceContext, double>,
    ops::UnsqueezeKernel<paddle::platform::CPUDeviceContext, int>,
    ops::UnsqueezeKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    unsqueeze_grad,
    ops::UnsqueezeGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::UnsqueezeGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::UnsqueezeGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::UnsqueezeGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
