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

class ElementwiseAddOpMaker : public ElementwiseOpMaker {
 protected:
  std::string GetName() const override { return "Add"; }
  std::string GetEquation() const override { return "Out = X + Y"; }

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
    return "Add two tensors element-wise";
  }
};

class ElementwiseAddDoubleGradDescMaker
    : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("elementwise_add_grad_grad");
    op->SetInput("Y", Input("Y"));
    op->SetInput("DOut", Input(framework::GradVarName("Out")));
    op->SetInput("DDX", OutputGrad(framework::GradVarName("X")));
    op->SetInput("DDY", OutputGrad(framework::GradVarName("Y")));

    op->SetAttrMap(Attrs());

    op->SetOutput("DDOut", InputGrad(framework::GradVarName("Out")));
    return op;
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_ELEMWISE_GRAD_MAKER(elementwise_add, Add);
REGISTER_ELEMWISE_EXPLICIT_OP_WITHOUT_GRAD(elementwise_add, Add);

namespace ops = paddle::operators;

REGISTER_OPERATOR(elementwise_add_grad, ops::ElementwiseOpExplicitGrad,
                  ops::ElementwiseGradOpInplace,
                  ops::ElementwiseGradNoBufVarsInference,
                  ops::ElementwiseAddDoubleGradDescMaker);
REGISTER_OPERATOR(elementwise_add_grad_grad,
                  ops::ElementwiseOpDoubleGradWithoutDXDY,
                  ops::ElementwiseDoubleGradOpInplace,
                  ops::ElementwiseDoubleGradNoBufVarsInference);

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
