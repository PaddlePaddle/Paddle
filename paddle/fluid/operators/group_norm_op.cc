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

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/backward.h"
#include "paddle/phi/infermeta/ternary.h"

namespace paddle {
namespace operators {

using Tensor = phi::DenseTensor;
using LoDTensor = framework::LoDTensor;
using DataLayout = framework::DataLayout;

class GroupNormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
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
          PADDLE_ENFORCE_EQ(epsilon >= 0.0f && epsilon <= 1.0f,
                            true,
                            platform::errors::InvalidArgument(
                                "'epsilon' in Op(GroupNorm) should be between"
                                "0.0 and 1.0f, But received [%s].",
                                epsilon));
        });
    AddAttr<int>("groups", "The number of groups that divided from channels.")
        .AddCustomChecker([](const int &groups) {
          PADDLE_ENFORCE_GT(
              groups,
              0,
              platform::errors::InvalidArgument(
                  "'groups' in Op(GroupNorm) should be greater than zero,"
                  "But received [%s].",
                  groups));
        });
    AddAttr<std::string>("data_layout",
                         "An optional string from: \"NHWC\", \"NCHW\". ")
        .SetDefault("NCHW");
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
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "GroupNormGrad");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "GroupNormGrad");
    OP_INOUT_CHECK(
        ctx->HasInput("Variance"), "Input", "Variance", "GroupNormGrad");
    OP_INOUT_CHECK(ctx->HasInput("Mean"), "Input", "Mean", "GroupNormGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Y")),
                   "Input",
                   framework::GradVarName("Y"),
                   "GroupNormGrad");

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

    PADDLE_ENFORCE_NOT_NULL(
        var,
        platform::errors::InvalidArgument(
            "Input(Y@GRAD) of GroupNormGradOp should not be null"));
    const Tensor *t = nullptr;
    if (var->IsType<Tensor>()) {
      t = &var->Get<Tensor>();
    } else if (var->IsType<LoDTensor>()) {
      t = &var->Get<LoDTensor>();
    }
    PADDLE_ENFORCE_NOT_NULL(
        t,
        platform::errors::InvalidArgument(
            "Input(Y@GRAD) Tensor of GroupNormGradOp should not be null"));
    return framework::OpKernelType(framework::TransToProtoVarType(t->dtype()),
                                   ctx.GetPlace());
  }
};

template <typename T>
class GroupNormGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> op) const override {
    op->SetType("group_norm_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Scale", this->Input("Scale"));
    op->SetInput("Bias", this->Input("Bias"));
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));
    op->SetInput("Y", this->Output("Y"));
    op->SetInput("Mean", this->Output("Mean"));
    op->SetInput("Variance", this->Output("Variance"));

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));
    op->SetOutput(framework::GradVarName("Scale"), this->InputGrad("Scale"));

    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_INPLACE_OP_INFERER(GroupNormGradInplaceInferer,
                           {framework::GradVarName("Y"),
                            framework::GradVarName("X")});

class GroupNormOpInferVarType
    : public framework::PassInDtypeAndVarTypeToOutput {
 protected:
  std::unordered_map<std::string, std::string> &GetInputOutputWithSameType()
      const override {
    static std::unordered_map<std::string, std::string> m{{"X", /*->*/ "Y"}};
    return m;
  }
};

}  // namespace operators
}  // namespace paddle

DECLARE_INFER_SHAPE_FUNCTOR(group_norm,
                            GroupNormInferShapeFunctor,
                            PD_INFER_META(phi::GroupNormInferMeta));

namespace ops = paddle::operators;
REGISTER_OPERATOR(group_norm,
                  ops::GroupNormOp,
                  ops::GroupNormOpMaker,
                  ops::GroupNormOpInferVarType,
                  ops::GroupNormGradMaker<paddle::framework::OpDesc>,
                  ops::GroupNormGradMaker<paddle::imperative::OpBase>,
                  GroupNormInferShapeFunctor);
REGISTER_OPERATOR(group_norm_grad,
                  ops::GroupNormGradOp,
                  ops::GroupNormGradInplaceInferer);
