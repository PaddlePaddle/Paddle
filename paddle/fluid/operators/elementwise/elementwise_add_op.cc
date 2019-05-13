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

#include "paddle/fluid/operators/elementwise/elementwise_add_op.h"
#include <memory>
#include <string>
#include "paddle/fluid/operators/elementwise/elementwise_op.h"

namespace paddle {
namespace operators {

class ElementwiseAddOpGradDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("elementwise_add_grad");
    op->SetInput("X", Input("X"));
    op->SetInput("Y", Input("Y"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetAttrMap(Attrs());
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), InputGrad("Y"));
    return op;
  }
};

class ElementwiseAddOpMaker : public ElementwiseOpMaker {
 protected:
  virtual std::string GetName() const { return "Add"; }
  virtual std::string GetEquation() const { return "Out = X + Y"; }
};

class ElementwiseAddDoubleGradDescMaker
    : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("elementwise_add_grad_grad");
    op->SetInput("X", Input("X"));
    op->SetInput("Y", Input("Y"));
    op->SetInput("DDX", OutputGrad(framework::GradVarName("X")));
    op->SetInput("DDY", OutputGrad(framework::GradVarName("Y")));

    op->SetAttrMap(Attrs());

    op->SetOutput("DDOut", InputGrad(framework::GradVarName("Out")));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), InputGrad("Y"));
    return op;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(elementwise_add, ops::ElementwiseOp,
                  ops::ElementwiseAddOpMaker, ops::ElementwiseOpInferVarType,
                  ops::ElementwiseAddOpGradDescMaker,
                  ops::ElementwiseOpInplace);
REGISTER_OPERATOR(elementwise_add_grad, ops::ElementwiseOpExplicitGrad,
                  ops::ElementwiseGradOpInplace,
                  ops::ElementwiseGradNoBufVarsInference,
                  ops::ElementwiseAddDoubleGradDescMaker);
REGISTER_OPERATOR(elementwise_add_grad_grad, ops::ElementwiseOpDoubleGrad);

REGISTER_OP_CPU_KERNEL(
    elementwise_add,
    ops::ElementwiseAddKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwiseAddKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwiseAddKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseAddKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    elementwise_add_grad,
    ops::ElementwiseAddGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwiseAddGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwiseAddGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseAddGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    elementwise_add_grad_grad,
    ops::ElementwiseAddDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        float>,
    ops::ElementwiseAddDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        double>,
    ops::ElementwiseAddDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        int>,
    ops::ElementwiseAddDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        int64_t>);
