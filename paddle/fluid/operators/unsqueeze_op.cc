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

    const auto& x_dims = ctx->GetInputDim("X");
    const auto& axes = ctx->Attrs().Get<std::vector<int>>("axes");
    // Check output tensor dims (<9).
    PADDLE_ENFORCE_LE(x_dims.size() + axes.size(), 9,
                      "Invalid dimnesions, dynamic dimensions must have "
                      "between [1, 9] dimensions.");
    // Check the range of unsqueeze aixs.
    for (int a : axes) {
      PADDLE_ENFORCE_LT(a, static_cast<int64_t>(x_dims.size() + axes.size()),
                        "The axis must be less than output tensor's rank.");
    }

    auto out_dims = GetOutputShape(axes, x_dims);
    ctx->SetOutputDim("Out", out_dims);
  }

  static framework::DDim GetOutputShape(const std::vector<int> unsqueeze_dims,
                                        const framework::DDim& in_dims) {
    int out_dims_size = in_dims.size() + unsqueeze_dims.size();
    bool should_unsqueeze[9] = {false};

    // Determines the dimensions should be unsqueezed in output tensor after.
    for (unsigned int idx = 0; idx < unsqueeze_dims.size(); ++idx) {
      int current = unsqueeze_dims[idx] < 0
                        ? unsqueeze_dims[idx] + out_dims_size
                        : unsqueeze_dims[idx];
      // Check current index.
      PADDLE_ENFORCE_GE(current, 0,
                        "Invaild axis, negative axis is out of range.");
      should_unsqueeze[idx] = true;
    }

    // Make output dimensions
    std::vector<int64_t> output_shape(out_dims_size, 0);
    for (int in_idx = 0, out_idx = 0; out_idx < out_dims_size; ++out_idx) {
      if (!should_unsqueeze[out_idx]) {
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
