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

#include "paddle/fluid/operators/elementwise/elementwise_mul_op.h"

#include <memory>
#include <string>

#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/prim/api/manual/backward/composite_backward_api.h"
#include "paddle/fluid/prim/utils/static/composite_grad_desc_maker.h"
#include "paddle/fluid/prim/utils/static/desc_tensor.h"

namespace paddle {
namespace operators {
class ElementwiseMulOpMaker : public ElementwiseOpMaker {
 protected:
  std::string GetName() const override { return "Mul"; }
  std::string GetEquation() const override { return "Out = X \\\\odot Y"; }

  void AddInputX() override {
    AddInput(
        "X",
        "(Variable), Tensor or phi::DenseTensor of any dimensions. Its dtype "
        "should be int32, int64, float32, float64.");
  }

  void AddInputY() override {
    AddInput(
        "Y",
        "(Variable), Tensor or phi::DenseTensor of any dimensions. Its dtype "
        "should be int32, int64, float32, float64.");
  }

  std::string GetOpFuntionality() const override {
    return "Multiply two tensors element-wise";
  }
};

template <typename T>
class ElementwiseMulOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("elementwise_mul_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
  }
};

class ElementwiseMulGradCompositeOpMaker
    : public prim::GradCompositeOpMakerBase {
  using prim::GradCompositeOpMakerBase::GradCompositeOpMakerBase;

 public:
  void Apply() override {
    auto x = this->GetSingleForwardInput("X");
    auto y = this->GetSingleForwardInput("Y");
    auto out_grad = this->GetSingleOutputGrad("Out");
    auto x_grad = this->GetSingleInputGrad("X");
    auto x_grad_p = this->GetOutputPtr(&x_grad);
    auto x_grad_name = this->GetOutputName(x_grad);
    auto y_grad = this->GetSingleInputGrad("Y");
    auto y_grad_p = this->GetOutputPtr(&y_grad);
    auto y_grad_name = this->GetOutputName(y_grad);
    prim::multiply_grad<prim::DescTensor>(
        x,
        y,
        out_grad,
        static_cast<int>(this->Attr<int>("axis")),
        x_grad_p,
        y_grad_p);
    VLOG(3) << "Runing mul_grad composite func";
    this->RecoverOutputName(x_grad, x_grad_name);
    this->RecoverOutputName(y_grad, y_grad_name);
  }
};

template <typename T>
class ElementwiseMulDoubleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("elementwise_mul_grad_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput("DOut", this->Input(framework::GradVarName("Out")));
    op->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
    op->SetInput("DDY", this->OutputGrad(framework::GradVarName("Y")));

    op->SetAttrMap(this->Attrs());

    op->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
  }
};

template <typename T>
class ElementwiseMulTripleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("elementwise_mul_triple_grad");
    // get input from double grad
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput("DOut", this->Input("DOut"));
    op->SetInput("DDX", this->Input("DDX"));
    op->SetInput("DDY", this->Input("DDY"));
    op->SetInput("D_DX", this->OutputGrad(framework::GradVarName("X")));
    op->SetInput("D_DY", this->OutputGrad(framework::GradVarName("Y")));
    op->SetInput("D_DDOut", this->OutputGrad("DDOut"));

    op->SetAttrMap(this->Attrs());

    // set outputs
    op->SetOutput("D_X", this->InputGrad("X"));
    op->SetOutput("D_Y", this->InputGrad("Y"));
    op->SetOutput("D_DOut", this->InputGrad("DOut"));
    op->SetOutput("D_DDX", this->InputGrad("DDX"));
    op->SetOutput("D_DDY", this->InputGrad("DDY"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(elementwise_mul,
                  ops::ElementwiseMulOp,
                  ops::ElementwiseMulOpMaker,
                  ops::ElementwiseOpInferVarType,
                  ops::ElementwiseMulOpGradMaker<paddle::framework::OpDesc>,
                  ops::ElementwiseMulOpGradMaker<paddle::imperative::OpBase>,
                  ops::ElementwiseMulGradCompositeOpMaker);
REGISTER_OPERATOR(
    elementwise_mul_grad,
    ops::ElementwiseOpGrad,
    ops::ElementwiseMulDoubleGradMaker<paddle::framework::OpDesc>,
    ops::ElementwiseMulDoubleGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(
    elementwise_mul_grad_grad,
    ops::ElementwiseOpDoubleGrad,
    ops::ElementwiseDoubleGradOpInplaceInferer,
    ops::ElementwiseMulTripleGradMaker<paddle::framework::OpDesc>,
    ops::ElementwiseMulTripleGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(elementwise_mul_triple_grad, ops::ElementwiseOpTripleGrad);

REGISTER_OP_VERSION(elementwise_mul)
    .AddCheckpoint(
        R"ROC(Register elementwise_mul for adding the attribute of Scale_y)ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "Scale_y",
            "In order to support the function of scaling the input Y when "
            "using the operator of elementwise_mul.",
            1.0f));
