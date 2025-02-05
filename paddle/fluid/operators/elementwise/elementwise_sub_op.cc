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
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

class ElementwiseSubOpMaker : public ElementwiseOpMaker {
 protected:
  std::string GetName() const override { return "Sub"; }
  std::string GetEquation() const override { return "Out = X - Y"; }

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
    return "Subtract two tensors element-wise";
  }
};

class ElementwiseSubCompositeGradOpMaker
    : public prim::CompositeGradOpMakerBase {
  using prim::CompositeGradOpMakerBase::CompositeGradOpMakerBase;

 public:
  void Apply() override {
    paddle::Tensor x = this->GetSingleForwardInput("X");
    paddle::Tensor y = this->GetSingleForwardInput("Y");
    paddle::Tensor out_grad = this->GetSingleOutputGrad("Out");
    paddle::Tensor dx = this->GetSingleInputGrad("X");
    auto dx_ptr = this->GetOutputPtr(&dx);
    std::string dx_name = this->GetOutputName(dx);
    paddle::Tensor dy = this->GetSingleInputGrad("Y");
    auto dy_ptr = this->GetOutputPtr(&dy);
    std::string dy_name = this->GetOutputName(dy);
    int axis = static_cast<int>(this->Attr<int>("axis"));
    PADDLE_ENFORCE_EQ(
        axis,
        -1,
        common::errors::InvalidArgument(
            "We only support axis = -1 in composite sub_grad but we got: ",
            axis));
    VLOG(6) << "Running sub_grad composite func";
    prim::subtract_grad<prim::DescTensor>(x, y, out_grad, axis, dx_ptr, dy_ptr);
    this->RecoverOutputName(dx, dx_name);
    this->RecoverOutputName(dy, dy_name);
  }
};

template <typename T>
class ElementwiseSubDoubleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("elementwise_sub_grad_grad");
    op->SetInput("Y", this->Input("Y"));
    op->SetInput("DOut", this->Input(framework::GradVarName("Out")));
    op->SetInput("DDX", this->OutputGrad(framework::GradVarName("X")));
    op->SetInput("DDY", this->OutputGrad(framework::GradVarName("Y")));

    op->SetAttrMap(this->Attrs());

    op->SetOutput("DDOut", this->InputGrad(framework::GradVarName("Out")));
  }
};

class ElementwiseSubCompositeDoubleGradOpMaker
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
                          "subtract_double_grad but we got: ",
                          axis));

    paddle::Tensor* grad_out_grad = this->GetOutputPtr(&grad_out_grad_t);
    std::string grad_out_grad_name = this->GetOutputName(grad_out_grad_t);

    VLOG(6) << "Running subtract_double_grad composite func";
    prim::subtract_double_grad<prim::DescTensor>(
        y, out_grad, ddx, ddy, axis, grad_out_grad);
    this->RecoverOutputName(grad_out_grad_t, grad_out_grad_name);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_ELEMWISE_GRAD_MAKER(elementwise_sub, Sub);

namespace ops = paddle::operators;

REGISTER_OPERATOR(elementwise_sub,
                  ::paddle::operators::ElementwiseOp,
                  ::paddle::operators::ElementwiseSubOpMaker,
                  ::paddle::operators::ElementwiseOpInferVarType,
                  elementwise_subGradMaker<::paddle::framework::OpDesc>,
                  elementwise_subGradMaker<::paddle::imperative::OpBase>,
                  ::paddle::operators::ElementwiseSubCompositeGradOpMaker,
                  ::paddle::operators::ElementwiseOpInplaceInferer);

REGISTER_OPERATOR(
    elementwise_sub_grad,
    ops::ElementwiseOpGrad,
    ops::ElementwiseGradOpInplaceInferer,
    ops::ElementwiseGradNoBufVarsInferer,
    ops::ElementwiseSubDoubleGradMaker<paddle::framework::OpDesc>,
    ops::ElementwiseSubDoubleGradMaker<paddle::imperative::OpBase>,
    ops::ElementwiseSubCompositeDoubleGradOpMaker);

REGISTER_OPERATOR(elementwise_sub_grad_grad,
                  ops::ElementwiseOpDoubleGradWithoutDXDY,
                  ops::ElementwiseDoubleGradOpInplaceInferer,
                  ops::ElementwiseDoubleGradNoBufVarsInferer);

REGISTER_OP_VERSION(elementwise_sub)
    .AddCheckpoint(
        R"ROC(Register elementwise_sub for adding the attribute of scale_y)ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "scale_y",
            "In order to support the function of scaling the input Y when "
            "using the operator of elementwise_sub.",
            1.0f));
