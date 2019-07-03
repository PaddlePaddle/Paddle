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
#include <memory>
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
    ctx->SetOutputDim("Out", ref_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   ctx.device_context());
  }
};

class ScatterGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("Updates"),
                      ctx->GetInputDim("Updates"));
    ctx->SetOutputDim(framework::GradVarName("X"),
                      ctx->GetInputDim(framework::GradVarName("Out")));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        ctx.Input<Tensor>(framework::GradVarName("Out"))->type(),
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
    AddAttr<bool>("overwrite",
                  "(bool, defalut: True) "
                  "The mode that updating the output when has same index,"
                  "If True, use the overwrite mode to update the output"
                  "of the same index, if False, use the accumulate mode to"
                  "update the output of the same index,Default value is True."
                  "You can set overwrite=False to implement scatter_add.")
        .SetDefault(true);
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

class ScatterGradDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("scatter_grad");
    op->SetInput("Ids", Input("Ids"));
    op->SetInput("Updates", Input("Updates"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetOutput(framework::GradVarName("Updates"), InputGrad("Updates"));
    op->SetAttrMap(Attrs());
    return op;
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERENCE(ScatterGradNoNeedBufferVarsInference,
                                      "Updates");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(scatter, ops::ScatterOp, ops::ScatterOpMaker,
                  ops::ScatterGradDescMaker);
REGISTER_OPERATOR(scatter_grad, ops::ScatterGradOp,
                  ops::ScatterGradNoNeedBufferVarsInference);
REGISTER_OP_CPU_KERNEL(scatter, ops::ScatterOpKernel<float>);
REGISTER_OP_CPU_KERNEL(scatter_grad, ops::ScatterGradientOpKernel<float>);
