/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/group_norm_op.h"
#include <string>

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using DataLayout = framework::DataLayout;

class GroupNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of GroupNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Y"),
                   "Output(Y) of GroupNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Mean"),
                   "Output(Mean) of GroupNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Variance"),
                   "Output(Variance) of GroupNormOp should not be null.");

    auto x_dim = ctx->GetInputDim("X");
    auto channel_num = x_dim[1];
    auto batch_size = x_dim[0];
    auto groups = ctx->Attrs().Get<int>("groups");
    PADDLE_ENFORCE_LE(
        groups, channel_num,
        "'groups' must be less equal than the number of channels.");
    PADDLE_ENFORCE_GE(groups, 1, "'groups' must be greater equal than 1.");

    if (ctx->HasInput("Scale")) {
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("Scale").size(), 1UL);
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("Scale")[0], channel_num);
    }
    if (ctx->HasInput("Bias")) {
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("Bias").size(), 1UL);
      PADDLE_ENFORCE_EQ(ctx->GetInputDim("Bias")[0], channel_num);
    }

    ctx->SetOutputDim("Y", ctx->GetInputDim("X"));
    ctx->SetOutputDim("Mean", {batch_size, groups});
    ctx->SetOutputDim("Variance", {batch_size, groups});
    ctx->ShareLoD("X", "Y");
  }
};

class GroupNormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor.");
    AddInput("Scale",
             "Scale is a 1-dimensional tensor of size C"
             "that is applied to the output.")
        .AsDispensable();
    AddInput("Bias",
             "Bias is a 1-dimensional tensor of size C "
             "that is applied to the output")
        .AsDispensable();
    AddOutput("Y", "Result after normalization.");
    AddOutput("Mean", "Mean of each group.").AsIntermediate();
    AddOutput("Variance", "Variance of each group.").AsIntermediate();

    AddAttr<float>("epsilon",
                   "Constant for numerical stability [default 1e-5].")
        .SetDefault(1e-5)
        .AddCustomChecker([](const float &epsilon) {
          PADDLE_ENFORCE(epsilon >= 0.0f && epsilon <= 1.0f,
                         "'epsilon' should be between 0.0 and 1.0.");
        });
    AddAttr<int>("groups", "The number of groups that divided from channels.")
        .AddCustomChecker([](const int &groups) {
          PADDLE_ENFORCE_GT(groups, 0, "'groups' should be greater than zero.");
        });

    AddComment(R"DOC(
Group Normalization

Refer to `Group Normalization <https://arxiv.org/abs/1803.08494>`_
)DOC");
  }
};

class GroupNormGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    // check input
    PADDLE_ENFORCE(ctx->HasInput("Y"),
                   "Input(Y) of GroupNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Mean"),
                   "Input(Mean) of GroupNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Variance"),
                   "Input(Variance) of GroupNormOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Y")),
                   "Input(Y@GRAD) of GroupNormOp should not be null.");

    // check output
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("Y"));
    }
    if (ctx->HasOutput(framework::GradVarName("Scale"))) {
      ctx->SetOutputDim(framework::GradVarName("Scale"),
                        ctx->GetInputDim("Scale"));
    }
    if (ctx->HasOutput(framework::GradVarName("Bias"))) {
      ctx->SetOutputDim(framework::GradVarName("Bias"),
                        ctx->GetInputDim("Bias"));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    const auto *var = ctx.InputVar(framework::GradVarName("Y"));
    if (var == nullptr) {
      PADDLE_THROW("can't find Y@GRAD");
    }
    const Tensor *t = nullptr;
    if (var->IsType<Tensor>()) {
      t = &var->Get<Tensor>();
    } else if (var->IsType<LoDTensor>()) {
      t = &var->Get<LoDTensor>();
    }
    if (t == nullptr) {
      PADDLE_THROW("can't find Y@GRAD");
    }
    return framework::OpKernelType(t->type(), ctx.GetPlace());
  }
};

class GroupNormGradMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *op = new framework::OpDesc();
    op->SetType("group_norm_grad");
    op->SetInput("Scale", Input("Scale"));
    op->SetInput("Bias", Input("Bias"));
    op->SetInput(framework::GradVarName("Y"), OutputGrad("Y"));
    op->SetInput("Y", Output("Y"));
    op->SetInput("Mean", Output("Mean"));
    op->SetInput("Variance", Output("Variance"));

    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetOutput(framework::GradVarName("Bias"), InputGrad("Bias"));
    op->SetOutput(framework::GradVarName("Scale"), InputGrad("Scale"));

    op->SetAttrMap(Attrs());

    return std::unique_ptr<framework::OpDesc>(op);
  }
};

class GroupNormInplaceInToOut : public framework::InplaceInToOut {
 public:
  using InplaceInToOut::InplaceInToOut;

 protected:
  std::unordered_map<std::string, std::string> Apply(
      const framework::OpDesc &op_desc,
      framework::BlockDesc *block) const override {
    return {{"X", "Y"}};
  }
};

class GroupNormGradInplaceInToOut : public framework::InplaceInToOut {
 public:
  using InplaceInToOut::InplaceInToOut;

 protected:
  std::unordered_map<std::string, std::string> Apply(
      const framework::OpDesc &op_desc,
      framework::BlockDesc *block) const override {
    return {{framework::GradVarName("Y"), framework::GradVarName("X")}};
  }
};

class GroupNormOpInferVarType
    : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string> GetInputOutputWithSameType()
      const override {
    return {{"X", /*->*/ "Y"}};
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(group_norm, ops::GroupNormOp, ops::GroupNormOpMaker,
                  ops::GroupNormOpInferVarType, ops::GroupNormGradMaker,
                  ops::GroupNormInplaceInToOut);
REGISTER_OPERATOR(group_norm_grad, ops::GroupNormGradOp,
                  ops::GroupNormGradInplaceInToOut);
REGISTER_OP_CPU_KERNEL(
    group_norm, ops::GroupNormKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GroupNormKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    group_norm_grad,
    ops::GroupNormGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GroupNormGradKernel<paddle::platform::CPUDeviceContext, double>);
