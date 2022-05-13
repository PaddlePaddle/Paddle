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

#include "paddle/fluid/operators/elementwise/elementwise_div_op.h"
#include <memory>
#include <string>

#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/platform/complex.h"

namespace paddle {
namespace operators {

class ElementwiseDivOpMaker : public ElementwiseOpMaker {
 protected:
  std::string GetName() const override { return "Div"; }
  std::string GetEquation() const override { return "Out = X / Y"; }

  void AddInputX() override {
    AddInput("X",
             "(Variable), Tensor or LoDTensor of any dimensions. Its dtype "
             "should be int32, int64, float32, float64.");
  }

  void AddInputY() override {
    AddInput("Y",
             "(Variable), Tensor or LoDTensor of any dimensions. Its dtype "
             "should be int32, int64, float32, float64.");
  }

  std::string GetOpFuntionality() const override {
    return "Divide two tensors element-wise";
  }
};

template <typename T>
class ElementwiseDivGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("elementwise_div_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    op->SetAttrMap(this->Attrs());
  }
};

template <typename T>
class ElementwiseDivDoubleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("elementwise_div_grad_grad");
    op->SetInput("Y", this->Input("Y"));
    op->SetInput("Out", this->Input("Out"));
    op->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
    op->SetInput("DDY", this->OutputGrad(framework::GradVarName("Y")));
    op->SetInput("DX", this->Output(framework::GradVarName("X")));

    op->SetAttrMap(this->Attrs());

    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    op->SetOutput("DOut", this->InputGrad("Out"));
    op->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(elementwise_div, ops::ElementwiseOp,
                  ops::ElementwiseDivOpMaker, ops::ElementwiseOpInferVarType,
                  ops::ElementwiseDivGradOpMaker<paddle::framework::OpDesc>,
                  ops::ElementwiseDivGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(
    elementwise_div_grad, ops::ElementwiseOpGrad,
    ops::ElementwiseDivDoubleGradMaker<paddle::framework::OpDesc>,
    ops::ElementwiseDivDoubleGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(elementwise_div_grad_grad, ops::ElementwiseDivOpDoubleGrad,
                  ops::ElementwiseDoubleGradOpInplaceInferer);

REGISTER_OP_VERSION(elementwise_div)
    .AddCheckpoint(
        R"ROC(Register elementwise_div for adding the attribute of Scale_y)ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "Scale_y",
            "In order to support the function of scaling the input Y when "
            "using the operator of elementwise_div.",
            1.0f));
