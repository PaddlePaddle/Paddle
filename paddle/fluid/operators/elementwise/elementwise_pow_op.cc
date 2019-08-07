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

#include "paddle/fluid/operators/elementwise/elementwise_pow_op.h"
#include <memory>
#include <string>
#include "paddle/fluid/operators/elementwise/elementwise_op.h"

namespace paddle {
namespace operators {

class ElementwisePowOpGradDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("elementwise_pow_grad");
    op->SetInput("X", Input("X"));
    op->SetInput("Y", Input("Y"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetAttrMap(Attrs());
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), InputGrad("Y"));
    return op;
  }
};
class ElementwisePowOpMaker : public ElementwiseOpMaker {
 protected:
  std::string GetName() const override { return "Pow"; }
  std::string GetEquation() const override { return "Out = X ^ Y"; }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(elementwise_pow, ops::ElementwiseOp,
                  ops::ElementwisePowOpMaker, ops::ElementwiseOpInferVarType,
                  ops::ElementwisePowOpGradDescMaker);
REGISTER_OPERATOR(elementwise_pow_grad, ops::ElementwiseOpGrad);

REGISTER_OP_CPU_KERNEL(
    elementwise_pow,
    ops::ElementwisePowKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwisePowKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwisePowKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwisePowKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    elementwise_pow_grad,
    ops::ElementwisePowGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwisePowGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwisePowGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwisePowGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
