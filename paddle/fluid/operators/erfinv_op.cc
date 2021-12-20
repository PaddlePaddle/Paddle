// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/erfinv_op.h"

namespace paddle {
namespace operators {

class ErfinvOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "erfinv");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "erfinv");

    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class ErfinvOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of erfinv op.");
    AddOutput("Out", "(Tensor), The output tensor of erfinv op.");
    AddComment(R"DOC(
Erfinv Operator.

This operator is used to compute inverse error function of input $X$.

The equation is:

$$erfinv(x) = {ndtri({x \over 2} + 0.5)} \over {\sqrt{2}}$$

The input `X` can carry the LoD (Level of Details) information,
or not. And the output shares the LoD information with input `X`.
)DOC");
  }
};

class ErfinvGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("Out"));
  }
};

template <typename T>
class ErfinvGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> op) const override {
    op->SetType("erfinv_grad");
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

DECLARE_INPLACE_OP_INFERER(ErfinvInplaceInferer, {"X", "Out"});

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(
    erfinv, paddle::operators::ErfinvOp, paddle::operators::ErfinvOpMaker,
    paddle::operators::ErfinvGradMaker<paddle::framework::OpDesc>,
    paddle::operators::ErfinvGradMaker<paddle::imperative::OpBase>,
    paddle::operators::ErfinvInplaceInferer);

REGISTER_OPERATOR(erfinv_grad, paddle::operators::ErfinvGradOp);

REGISTER_OP_CPU_KERNEL(
    erfinv,
    paddle::operators::ErfinvKernel<paddle::platform::CPUDeviceContext, float>,
    paddle::operators::ErfinvKernel<paddle::platform::CPUDeviceContext,
                                    double>);

REGISTER_OP_CPU_KERNEL(
    erfinv_grad,
    paddle::operators::ErfinvGradKernel<paddle::platform::CPUDeviceContext,
                                        float>,
    paddle::operators::ErfinvGradKernel<paddle::platform::CPUDeviceContext,
                                        double>);
