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

template <typename T>
class ElementwisePowOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("elementwise_pow_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
  }
};
class ElementwisePowOpMaker : public ElementwiseOpMaker {
 protected:
  std::string GetName() const override { return "Pow"; }
  std::string GetEquation() const override { return "Out = X ^ Y"; }

  void AddInputX() override { AddInput("X", "(Variable), The Base."); }

  void AddInputY() override { AddInput("Y", "(Variable), The exponents."); }

  std::string GetOpFuntionality() const override {
    return "First tensor elements raised to powers from the second tensor, "
           "element-wise.";
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(elementwise_pow, ops::ElementwiseOp,
                  ops::ElementwisePowOpMaker, ops::ElementwiseOpInferVarType,
                  ops::ElementwisePowOpGradMaker<paddle::framework::OpDesc>,
                  ops::ElementwisePowOpGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(elementwise_pow_grad, ops::ElementwiseOpGrad);

REGISTER_OP_VERSION(elementwise_pow)
    .AddCheckpoint(
        R"ROC(Register elementwise_pow for adding the attribute of Scale_y)ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "Scale_y",
            "In order to support the function of scaling the input Y when "
            "using the operator of elementwise_pow.",
            1.0f));
