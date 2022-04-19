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

namespace paddle {
namespace framework {
class OpDesc;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
namespace platform {
class CPUDeviceContext;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

class ElementwiseSubOpMaker : public ElementwiseOpMaker {
 protected:
  std::string GetName() const override { return "Sub"; }
  std::string GetEquation() const override { return "Out = X - Y"; }

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
    return "Substract two tensors element-wise";
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

}  // namespace operators
}  // namespace paddle

REGISTER_ELEMWISE_GRAD_MAKER(elementwise_sub, Sub);

namespace ops = paddle::operators;

REGISTER_OPERATOR(elementwise_sub, ::paddle::operators::ElementwiseOp,
                  ::paddle::operators::ElementwiseSubOpMaker,
                  ::paddle::operators::ElementwiseOpInferVarType,
                  elementwise_subGradMaker<::paddle::framework::OpDesc>,
                  elementwise_subGradMaker<::paddle::imperative::OpBase>,
                  ::paddle::operators::ElementwiseOpInplaceInferer);

REGISTER_OPERATOR(
    elementwise_sub_grad, ops::ElementwiseOpGrad,
    ops::ElementwiseGradOpInplaceInferer, ops::ElementwiseGradNoBufVarsInferer,
    ops::ElementwiseSubDoubleGradMaker<paddle::framework::OpDesc>,
    ops::ElementwiseSubDoubleGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(elementwise_sub_grad_grad,
                  ops::ElementwiseOpDoubleGradWithoutDXDY,
                  ops::ElementwiseDoubleGradOpInplaceInferer,
                  ops::ElementwiseDoubleGradNoBufVarsInferer);

REGISTER_OP_VERSION(elementwise_sub)
    .AddCheckpoint(
        R"ROC(Register elementwise_sub for adding the attribute of Scale_y)ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "Scale_y",
            "In order to support the function of scaling the input Y when "
            "using the operator of elementwise_sub.",
            1.0f));
