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

#include <string>

#include "paddle/fluid/operators/elementwise/elementwise_op.h"
#include "paddle/fluid/prim/api/composite_backward/composite_backward_api.h"
#include "paddle/fluid/prim/utils/static/composite_grad_desc_maker.h"
#include "paddle/fluid/prim/utils/static/desc_tensor.h"
namespace paddle {
namespace framework {
class OpDesc;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {

class ElementwiseAddOpMaker : public ElementwiseOpMaker {
 protected:
  std::string GetName() const override { return "Add"; }
  std::string GetEquation() const override { return "Out = X + Y"; }

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

  std::string GetOpFunctionality() const override {
    return "Add two tensors element-wise";
  }
};

class ElementwiseAddCompositeGradOpMaker
    : public prim::CompositeGradOpMakerBase {
  using prim::CompositeGradOpMakerBase::CompositeGradOpMakerBase;

 public:
  void Apply() override {
    paddle::Tensor x = this->GetSingleForwardInput("X");
    paddle::Tensor y = this->GetSingleForwardInput("Y");
    paddle::Tensor out_grad = this->GetSingleOutputGrad("Out");
    paddle::Tensor dx = this->GetSingleInputGrad("X");
    auto* dx_ptr = this->GetOutputPtr(&dx);
    std::string dx_name = this->GetOutputName(dx);
    paddle::Tensor dy = this->GetSingleInputGrad("Y");
    auto* dy_ptr = this->GetOutputPtr(&dy);
    std::string dy_name = this->GetOutputName(dy);
    int axis = static_cast<int>(this->Attr<int>("axis"));
    PADDLE_ENFORCE_EQ(
        axis,
        -1,
        common::errors::InvalidArgument(
            "We only support axis = -1 in composite add_grad but we got: ",
            axis));
    VLOG(6) << "Running add_grad composite func";
    prim::add_grad<prim::DescTensor>(x, y, out_grad, axis, dx_ptr, dy_ptr);
    this->RecoverOutputName(dx, dx_name);
    this->RecoverOutputName(dy, dy_name);
  }
};

template <typename T>
class ElementwiseAddDoubleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("elementwise_add_grad_grad");
    op->SetInput("Y", this->Input("Y"));
    op->SetInput("DOut", this->Input(framework::GradVarName("Out")));
    op->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
    op->SetInput("DDY", this->OutputGrad(framework::GradVarName("Y")));

    op->SetAttrMap(this->Attrs());

    op->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
  }
};

class ElementwiseAddCompositeDoubleGradOpMaker
    : public prim::CompositeGradOpMakerBase {
  using prim::CompositeGradOpMakerBase::CompositeGradOpMakerBase;

 public:
  void Apply() override {
    // get input
    paddle::Tensor y = this->GetSingleForwardInput("Y");
    paddle::Tensor out_grad = this->GetSingleOutputGrad("Out");
    paddle::optional<paddle::Tensor> ddx =
        this->GetOptionalSingleOutputGrad(framework::GradVarName("X"));
    paddle::optional<paddle::Tensor> ddy =
        this->GetOptionalSingleOutputGrad(framework::GradVarName("Y"));
    // get output
    paddle::Tensor grad_out_grad_t =
        this->GetSingleInputGrad(framework::GradVarName("Out"));

    // get attr
    int axis = static_cast<int>(this->Attr<int>("axis"));
    PADDLE_ENFORCE_EQ(axis,
                      -1,
                      common::errors::InvalidArgument(
                          "We only support axis = -1 in composite "
                          "add_double_grad but we got: ",
                          axis));

    paddle::Tensor* grad_out_grad = this->GetOutputPtr(&grad_out_grad_t);
    std::string grad_out_grad_name = this->GetOutputName(grad_out_grad_t);

    VLOG(6) << "Running add_double_grad composite func";
    prim::add_double_grad<prim::DescTensor>(
        y, out_grad, ddx, ddy, axis, grad_out_grad);
    this->RecoverOutputName(grad_out_grad_t, grad_out_grad_name);
  }
};

template <typename T>
class ElementwiseAddTripleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("elementwise_add_triple_grad");
    op->SetInput("DDX", this->Input("DDX"));
    op->SetInput("DDY", this->Input("DDY"));
    op->SetInput("D_DDOut", this->OutputGrad("DDOut"));

    op->SetAttrMap(this->Attrs());

    op->SetOutput("D_DDX", this->InputGrad("DDX"));
    op->SetOutput("D_DDY", this->InputGrad("DDY"));
  }
};

class ElementwiseAddCompositeTripleGradOpMaker
    : public prim::CompositeGradOpMakerBase {
  using prim::CompositeGradOpMakerBase::CompositeGradOpMakerBase;

 public:
  void Apply() override {
    // get input
    paddle::Tensor ddx = this->GetSingleForwardInput("DDX");
    paddle::Tensor ddy = this->GetSingleForwardInput("DDY");
    paddle::Tensor d_ddout = this->GetSingleOutputGrad("DDOut");

    // get output
    paddle::Tensor grad_grad_x_t =
        this->GetSingleInputGrad(framework::GradVarName("DDX"));
    paddle::Tensor grad_grad_y_t =
        this->GetSingleInputGrad(framework::GradVarName("DDY"));
    // get attr
    int axis = static_cast<int>(this->Attr<int>("axis"));
    PADDLE_ENFORCE_EQ(axis,
                      -1,
                      common::errors::InvalidArgument(
                          "We only support axis = -1 in composite "
                          "add_triple_grad but we got: ",
                          axis));

    paddle::Tensor* grad_grad_x = this->GetOutputPtr(&grad_grad_x_t);
    std::string grad_grad_x_name = this->GetOutputName(grad_grad_x_t);
    paddle::Tensor* grad_grad_y = this->GetOutputPtr(&grad_grad_y_t);
    std::string grad_grad_y_name = this->GetOutputName(grad_grad_y_t);

    VLOG(6) << "Running add_triple_grad composite func";
    prim::add_triple_grad<prim::DescTensor>(
        ddx, ddy, d_ddout, axis, grad_grad_x, grad_grad_y);
    this->RecoverOutputName(grad_grad_x_t, grad_grad_x_name);
    this->RecoverOutputName(grad_grad_y_t, grad_grad_y_name);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_ELEMWISE_GRAD_MAKER(elementwise_add, Add);
REGISTER_OPERATOR(elementwise_add,
                  ::paddle::operators::ElementwiseOp,
                  ::paddle::operators::ElementwiseAddOpMaker,
                  ::paddle::operators::ElementwiseOpInferVarType,
                  elementwise_addGradMaker<::paddle::framework::OpDesc>,
                  elementwise_addGradMaker<::paddle::imperative::OpBase>,
                  ::paddle::operators::ElementwiseAddCompositeGradOpMaker,
                  ::paddle::operators::ElementwiseOpInplaceInferer);

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    elementwise_add_grad,
    ops::ElementwiseOpGrad,
    ops::ElementwiseGradOpInplaceInferer,
    ops::ElementwiseGradNoBufVarsInferer,
    ops::ElementwiseAddDoubleGradMaker<paddle::framework::OpDesc>,
    ops::ElementwiseAddDoubleGradMaker<paddle::imperative::OpBase>,
    ops::ElementwiseAddCompositeDoubleGradOpMaker);

REGISTER_OPERATOR(
    elementwise_add_grad_grad,
    ops::ElementwiseOpDoubleGradWithoutDXDY,
    ops::ElementwiseDoubleGradOpInplaceInferer,
    ops::ElementwiseDoubleGradNoBufVarsInferer,
    ops::ElementwiseAddTripleGradMaker<paddle::framework::OpDesc>,
    ops::ElementwiseAddTripleGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(elementwise_add_triple_grad,
                  ops::ElementwiseOpTripleGrad,
                  ops::ElementwiseTripleGradOpInplaceInferer,
                  ops::ElementwiseTripleGradNoBufVarsInferer);

// A specialization elementwise_add operator, used in gradient accumulation with
// inplace addto.
REGISTER_OPERATOR(
    grad_add,
    paddle::operators::ElementwiseOp,
    paddle::operators::ElementwiseAddOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_VERSION(elementwise_add)
    .AddCheckpoint(
        R"ROC(Register elementwise_add for adding the attribute of
       Scale_y)ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "Scale_y",
            "In order to support the function of scaling the input Y when "
            "using the operator of elementwise_add.",
            1.0f));
