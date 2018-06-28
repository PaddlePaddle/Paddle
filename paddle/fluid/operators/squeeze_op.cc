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

#include "paddle/fluid/operators/squeeze_op.h"
#include <string>
#include <vector>

namespace paddle {
namespace operators {

using framework::OpKernelType;
using framework::Tensor;

class SqueezeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SqueezeOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SqueezeOp should not be null.");

    const auto& x_dims = ctx->GetInputDim("X");
    // TODO(chenweihang): need check input tensor dims (<9).

    const auto& axes = ctx->Attrs().Get<std::vector<int>>("axes");
    // TODO(chenweihang): need check axes is valid.
    // PADDLE_ENFORCE();
    for (int a : axes) {
      PADDLE_ENFORCE_LT(a, x_dims.size(),
                        "The axis must be less than input tensor's rank.");
    }

    auto out_dims = GetOutputShape(axes, x_dims);
    ctx->SetOutputDim("Out", out_dims);
    // TODO(chenweihang): need other check.
  }

  static framework::DDim GetOutputShape(const std::vector<int> squeeze_dims,
                                        const framework::DDim& in_dims) {
    int num_squeeze_dims = squeeze_dims.size();
    int cnt_squeezed_dims = 0;
    bool should_squeeze[9] = {false};

    // Determines number of dimensions of output tensor after squeeze.
    // Mark and count the dimensions need to be squeezed
    if (num_squeeze_dims == 0) {
      for (int idx = 0; idx < in_dims.size(); ++idx) {
        if (in_dims[idx] == 1) {
          should_squeeze[idx] = true;
          ++cnt_squeezed_dims;
        }
      }
    } else {
      for (int idx = 0; idx < num_squeeze_dims; ++idx) {
        int current = squeeze_dims[idx] < 0 ? squeeze_dims[idx] + in_dims.size()
                                            : squeeze_dims[idx];
        // TODO(chenweihang): shoude use PADALE_ENFORCE ? or if.
        PADDLE_ENFORCE_GE(current, 0, "Invalid axis is given.");
        PADDLE_ENFORCE_LT(current, in_dims.size(), "Invalid axis is given.");
        PADDLE_ENFORCE_EQ(in_dims[current], 1, "Invalid axis is given.");

        if (!(should_squeeze[current])) ++cnt_squeezed_dims;
        should_squeeze[current] = true;
      }
    }

    // Make output dimensions
    std::vector<int64_t> output_shape(in_dims.size() - cnt_squeezed_dims, 0);
    for (int in_idx = 0, out_idx = 0; in_idx < in_dims.size(); ++in_idx) {
      if (!should_squeeze[in_idx]) {
        output_shape[out_idx++] = in_dims[in_idx];
      }
    }

    return framework::make_ddim(output_shape);
  }
};

class SqueezeOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), Tensors with at least max(dims) dimensions.");
    AddOutput("Out", "(Tensor), Reshaped tensor with same data as input.");
    AddAttr<std::vector<int>>("axes",
                              "List of positive integers,"
                              " indicate the dimensions to squeeze.");
    AddAttr<bool>("inplace",
                  "(default: false) Change the source tensor's shape without "
                  "memory copy. When Attr(inplace) is set true, the output "
                  "tensor shares memory with Input(X), otherwise, a new output "
                  "tensor is created, and its data are copied from Input(x).")
        .SetDefault(false);
    AddComment(R"DOC(
		Squeeze Operator.
		
		Remove single-dimensional entries from the shape of a tensor. 
		Takes a parameter axes with a list of axes to squeeze. 
		If axes is not provided, all the single dimensions will be removed from the shape. 
        If an axis is selected with shape entry not equal to one, an error is raised.
    )DOC");
  }
};

class SqueezeGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SqueezeOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Output(Out@GRAD/) of SqueezeOp should not be null.");
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
REGISTER_OPERATOR(squeeze, ops::SqueezeOp, ops::SqueezeOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(squeeze_grad, ops::SqueezeGradOp);
REGISTER_OP_CPU_KERNEL(
    squeeze, ops::SqueezeKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SqueezeKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SqueezeKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SqueezeKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    squeeze_grad,
    ops::SqueezeGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SqueezeGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SqueezeGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SqueezeGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
