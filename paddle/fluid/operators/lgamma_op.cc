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

#include "paddle/fluid/operators/lgamma_op.h"

namespace paddle {
namespace operators {

class LgammaOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of lgamma op.");
    AddOutput("Out", "(Tensor), The output tensor of lgamma op.");
    AddComment(R"DOC(
Lgamma Operator.

This operator performs elementwise lgamma for input $X$.
$$out = log\Gamma(x)$$

)DOC");
  }
};

class LgammaOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Lgamma");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Lgamma");

    auto in_dims = ctx->GetInputDim("X");

    ctx->SetOutputDim("Out", in_dims);
    ctx->ShareLoD("X", "Out");
  }
};

template <typename T>
class LgammaGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("lgamma_grad");
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetInput("X", this->Input("X"));
    retv->SetAttrMap(this->Attrs());
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

class LgammaGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@Grad", "LgammaGrad");
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "LgammaGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                   "X@Grad", "LgammaGrad");

    auto dout_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    ctx->SetOutputDim(framework::GradVarName("X"), dout_dims);
    ctx->ShareLoD(framework::GradVarName("Out"), framework::GradVarName("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(lgamma, ops::LgammaOp, ops::LgammaOpMaker,
                  ops::LgammaGradMaker<paddle::framework::OpDesc>,
                  ops::LgammaGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(lgamma_grad, ops::LgammaGradOp);

REGISTER_OP_CPU_KERNEL(
    lgamma, ops::LgammaKernel<paddle::platform::CPUDeviceContext, float>,
    ops::LgammaKernel<paddle::platform::CPUDeviceContext, double>)

REGISTER_OP_CPU_KERNEL(
    lgamma_grad,
    ops::LgammaGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::LgammaGradKernel<paddle::platform::CPUDeviceContext, double>);
