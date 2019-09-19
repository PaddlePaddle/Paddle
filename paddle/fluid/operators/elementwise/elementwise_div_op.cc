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

namespace paddle {
namespace operators {

class ElementwiseDivOpMaker : public ElementwiseOpMaker {
 protected:
  std::string GetName() const override { return "Div"; }
  std::string GetEquation() const override { return "Out = X / Y"; }
};

template <typename T>
class ElementwiseDivGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  std::unique_ptr<T> Apply() const override {
    std::unique_ptr<T> op(new T());
    op->SetType("elementwise_div_grad");
    op->SetInput("Y", this->Input("Y"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
    op->SetAttrMap(this->Attrs());
    return op;
  }
};

template <typename T>
class ElementwiseDivDoubleGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  std::unique_ptr<T> Apply() const override {
    std::unique_ptr<T> op(new T());
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

    return op;
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
REGISTER_OPERATOR(elementwise_div_grad_grad, ops::ElementwiseDivOpDoubleGrad);

REGISTER_OP_CPU_KERNEL(
    elementwise_div,
    ops::ElementwiseDivKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwiseDivKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwiseDivKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseDivKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    elementwise_div_grad,
    ops::ElementwiseDivGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwiseDivGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwiseDivGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseDivGradKernel<paddle::platform::CPUDeviceContext, int64_t>);

REGISTER_OP_CPU_KERNEL(
    elementwise_div_grad_grad,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        float>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        double>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        int>,
    ops::ElementwiseDivDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        int64_t>);
