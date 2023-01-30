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
<<<<<<< HEAD
#include "paddle/fluid/prim/api/manual/backward/composite_backward_api.h"
#include "paddle/fluid/prim/utils/static/composite_grad_desc_maker.h"
#include "paddle/fluid/prim/utils/static/desc_tensor.h"
=======

>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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
<<<<<<< HEAD
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
=======
    AddInput("X",
             "(Variable), Tensor or LoDTensor of any dimensions. Its dtype "
             "should be int32, int64, float32, float64.");
  }

  void AddInputY() override {
    AddInput("Y",
             "(Variable), Tensor or LoDTensor of any dimensions. Its dtype "
             "should be int32, int64, float32, float64.");
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
  }

  std::string GetOpFuntionality() const override {
    return "Add two tensors element-wise";
  }
};

<<<<<<< HEAD
class ElementwiseAddGradCompositeOpMaker
    : public prim::GradCompositeOpMakerBase {
  using prim::GradCompositeOpMakerBase::GradCompositeOpMakerBase;

 public:
  void Apply() override {
    paddle::experimental::Tensor x = this->GetSingleForwardInput("X");
    paddle::experimental::Tensor y = this->GetSingleForwardInput("Y");
    paddle::experimental::Tensor out_grad = this->GetSingleOutputGrad("Out");
    paddle::experimental::Tensor dx = this->GetSingleInputGrad("X");
    auto dx_ptr = this->GetOutputPtr(&dx);
    std::string dx_name = this->GetOutputName(dx);
    paddle::experimental::Tensor dy = this->GetSingleInputGrad("Y");
    auto dy_ptr = this->GetOutputPtr(&dy);
    std::string dy_name = this->GetOutputName(dy);
    int axis = static_cast<int>(this->Attr<int>("axis"));
    VLOG(3) << "Runing add_grad composite func";
    prim::add_grad<prim::DescTensor>(x, y, out_grad, axis, dx_ptr, dy_ptr);
    this->RecoverOutputName(dx, dx_name);
    this->RecoverOutputName(dy, dy_name);
  }
};

=======
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
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

}  // namespace operators
}  // namespace paddle

REGISTER_ELEMWISE_GRAD_MAKER(elementwise_add, Add);
<<<<<<< HEAD
REGISTER_OPERATOR(elementwise_add,
                  ::paddle::operators::ElementwiseOp,
                  ::paddle::operators::ElementwiseAddOpMaker,
                  ::paddle::operators::ElementwiseOpInferVarType,
                  elementwise_addGradMaker<::paddle::framework::OpDesc>,
                  elementwise_addGradMaker<::paddle::imperative::OpBase>,
                  ::paddle::operators::ElementwiseAddGradCompositeOpMaker,
                  ::paddle::operators::ElementwiseOpInplaceInferer);

namespace ops = paddle::operators;

=======
REGISTER_ELEMWISE_EXPLICIT_OP_WITHOUT_GRAD(elementwise_add, Add);

namespace ops = paddle::operators;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
REGISTER_OPERATOR(
    elementwise_add_grad,
    ops::ElementwiseOpGrad,
    ops::ElementwiseGradOpInplaceInferer,
    ops::ElementwiseGradNoBufVarsInferer,
    ops::ElementwiseAddDoubleGradMaker<paddle::framework::OpDesc>,
    ops::ElementwiseAddDoubleGradMaker<paddle::imperative::OpBase>);

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
