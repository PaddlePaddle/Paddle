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

#include "paddle/fluid/operators/scatter_op.h"
#include "paddle/fluid/framework/ddim.h"

namespace paddle {
namespace operators {

class ScatterOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of ScatterOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Ids"),
                   "Input(Ids) of ScatterOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Updates"),
                   "Input(Updates) of ScatterOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ScatterOp should not be null.");

    auto updates_dims = ctx->GetInputDim("Updates");
    auto ref_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Ids").size(), 1,
                      "Update Ids should be 1-D.");
    PADDLE_ENFORCE_EQ(ref_dims.size(), updates_dims.size(),
                      "Xerence and Updates should have the same shape size");
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Updates")[0],
                      ctx->GetInputDim("Ids")[0],
                      "Updates and Ids should have same batch-size.");
    framework::DDim data_dim(updates_dims);
    for (int i = 1; i < data_dim.size(); ++i) {
      PADDLE_ENFORCE_EQ(data_dim[i], updates_dims[i]);
    }
    ctx->SetOutputDim("Out", ref_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("X")->type()),
        ctx.device_context());
  }
};

class ScatterGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("Updates"),
                      ctx->GetInputDim("Updates"));
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("X")->type()),
        ctx.device_context());
  }
};

class ScatterOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The source input of scatter op");
    AddInput("Ids", "The index input of scatter op where X will be updated");
    AddInput("Updates", "The updated value of scatter op");
    AddOutput("Out", "The output of scatter op");
    AddComment(R"DOC(
Scatter Operator.

This operator obtains output by updating the input on selected indices on the first axis:

$$
Out = X \\
Out[Ids] = Updates
$$

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(scatter, ops::ScatterOp, ops::ScatterOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(scatter_grad, ops::ScatterGradOp);
REGISTER_OP_CPU_KERNEL(scatter, ops::ScatterOpKernel<float>);
REGISTER_OP_CPU_KERNEL(scatter_grad, ops::ScatterGradientOpKernel<float>);
