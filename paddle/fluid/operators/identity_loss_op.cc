/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/unary.h"

namespace paddle {
namespace operators {

class IdentityLossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        platform::CPUPlace());
  }
};

class IdentityLossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The input of identity_loss op");
    AddOutput("Out", "(Tensor) The output of identity_loss op");
    AddAttr<int>("reduction", "(int, default 1). The reduction.")
        .SetDefault(1)
        .InEnum({0, 1, 2});
    AddComment(R"DOC(
IdentityLoss Operator mark the Loss var.

)DOC");
  }
};

class IdentityLossGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    ctx->ShareLoD("X", framework::GradVarName("X"));
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto input_data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return framework::OpKernelType(input_data_type, platform::CPUPlace());
  }
};

template <typename T>
class IdentityLossGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("identity_loss_grad");
    grad_op->SetInput("X", this->Input("X"));
    grad_op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    grad_op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

DECLARE_INPLACE_OP_INFERER(IdentityLossInplaceInferer, {"X", "Out"});
DECLARE_INPLACE_OP_INFERER(IdentityLossGradInplaceInferer,
                           {framework::GradVarName("Out"),
                            framework::GradVarName("X")});

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
DECLARE_INFER_SHAPE_FUNCTOR(identity_loss,
                            IdentityLossInferShapeFunctor,
                            PD_INFER_META(phi::IdentityLossInferMeta));

REGISTER_OPERATOR(identity_loss,
                  ops::IdentityLossOp,
                  ops::IdentityLossOpMaker,
                  ops::IdentityLossGradMaker<paddle::framework::OpDesc>,
                  ops::IdentityLossGradMaker<paddle::imperative::OpBase>,
                  ops::IdentityLossInplaceInferer,
                  IdentityLossInferShapeFunctor);

REGISTER_OPERATOR(identity_loss_grad,
                  ops::IdentityLossGradOp,
                  ops::IdentityLossGradInplaceInferer);
