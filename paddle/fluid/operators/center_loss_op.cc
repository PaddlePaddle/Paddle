/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/center_loss_op.h"

#include <memory>
#include <string>

namespace paddle {
namespace operators {
class CenterLossOp : public framework::OperatorWithKernel {
 public:
  CenterLossOp(const std::string &type,
               const framework::VariableNameMap &inputs,
               const framework::VariableNameMap &outputs,
               const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "CenterLoss");
    auto x_dims = ctx->GetInputDim("X");

    OP_INOUT_CHECK(ctx->HasInput("CenterUpdateRate"),
                   "Input",
                   "CenterUpdateRate",
                   "CenterLoss");
    OP_INOUT_CHECK(ctx->HasInput("Label"), "Input", "Label", "CenterLoss");
    OP_INOUT_CHECK(ctx->HasInput("Centers"), "Input", "Centers", "CenterLoss");
    OP_INOUT_CHECK(ctx->HasOutput("SampleCenterDiff"),
                   "Output",
                   "SampleCenterDiff",
                   "CenterLoss");
    OP_INOUT_CHECK(ctx->HasOutput("Loss"), "Output", "Loss", "CenterLoss");
    OP_INOUT_CHECK(
        ctx->HasOutput("CentersOut"), "Output", "CentersOut", "CenterLoss");

    ctx->SetOutputDim("SampleCenterDiff",
                      {x_dims[0], product(x_dims) / x_dims[0]});
    ctx->SetOutputDim("CentersOut", ctx->GetInputDim("Centers"));
    ctx->SetOutputDim("Loss", {x_dims[0], 1});
    ctx->ShareLoD("X", /*->*/ "Loss");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class CenterLossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Input tensor of center_loss operator.");
    AddInput("Label", "(Tensor) Input tensor of center_loss operator.");
    AddInput("Centers", "(Tensor) Input tensor of center_loss operator.");
    AddInput("CenterUpdateRate",
             "(Tensor) Input tensor of center_loss operator.");

    AddOutput("CentersOut", "(Tensor) Input tensor of center_loss operator.");
    AddOutput("SampleCenterDiff",
              "(Tensor) output tensor of center_loss operator.");
    AddOutput("Loss", "(Tensor) Output tensor of center_loss operator.");

    AddAttr<int>("cluster_num",
                 "The output cluster num of the center_loss operator.");
    AddAttr<bool>("need_update", "whether need to update center info.");
    AddComment(R"DOC(
**CenterLoss operator**
implemention of the center loss function in the papper<<A Discriminative
Feature Learning Approach for Deep Face Recognition>>, equations in this  implement
is:loss = 1/2 * (x-y)^2 ,where x(X) means the deep feature(output of last hidden layer )
and y(Label) the target label
)DOC");
  }
};

class CenterLossGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("SampleCenterDiff"),
                   "Input",
                   "SampleCenterDiff",
                   "CenterLossGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Loss")),
                   "Input",
                   framework::GradVarName("Loss"),
                   "CenterLossGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")),
                   "Output",
                   framework::GradVarName("X"),
                   "CenterLossGrad");

    auto x_dims = ctx->GetInputDim("X");
    auto x_grad_name = framework::GradVarName("X");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "SampleCenterDiff"),
        ctx.device_context());
  }
};

template <typename T>
class CenterLossOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("center_loss_grad");
    retv->SetInput(framework::GradVarName("Loss"), this->OutputGrad("Loss"));
    retv->SetInput("SampleCenterDiff", this->Output("SampleCenterDiff"));
    retv->SetInput("X", this->Input("X"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));

    retv->SetAttrMap(this->Attrs());
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(CenterLossGradNoNeedBufVarsInferer, "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPUCtx = phi::CPUContext;

REGISTER_OPERATOR(center_loss,
                  ops::CenterLossOp,
                  ops::CenterLossOpMaker,
                  ops::CenterLossOpGradMaker<paddle::framework::OpDesc>,
                  ops::CenterLossOpGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(center_loss_grad,
                  ops::CenterLossGradOp,
                  ops::CenterLossGradNoNeedBufVarsInferer);

REGISTER_OP_CPU_KERNEL(center_loss,
                       ops::CenterLossKernel<CPUCtx, float>,
                       ops::CenterLossKernel<CPUCtx, double>);

REGISTER_OP_CPU_KERNEL(center_loss_grad,
                       ops::CenterLossGradKernel<CPUCtx, float>,
                       ops::CenterLossGradKernel<CPUCtx, double>);
