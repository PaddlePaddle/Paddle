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
#include "paddle/fluid/operators/elementwise/elementwise_op.h"

#include <vector>
#include <string>
#include <memory>

namespace paddle{
namespace operators{

class ElementwiseAddGradGradMaker: public framework::GradOpDescMakerBase{
 public: 
  using framework::GradOpDescMakerBase::GradOpDescMakerBase;

  std::vector<std::unique_ptr<framework::OpDesc>> operator()() const override{
    std::vector<std::unique_ptr<framework::OpDesc>> ops;
    auto ddy = OutputGrad(framework::GradVarName("Y"));
    auto ddx = OutputGrad(framework::GradVarName("X"));
    auto x_grads = InputGrad("X"); 
    if(!x_grads.empty()){
      if(!ddy.empty()){
        auto* x_grad_op = new framework::OpDesc();
        x_grad_op->SetType("scale");
        x_grad_op->SetInput("X", OutputGrad(framework::GradVarName("Y")));
        x_grad_op->SetOutput("Out", x_grads);
        x_grad_op->SetAttr("scale", 0.0f);
        ops.emplace_back(x_grad_op);
      }
    }

    auto y_grads = InputGrad("Y");
    if(!y_grads.empty()){
      if(!ddx.empty()){
        auto* y_grad_op = new framework::OpDesc();
        y_grad_op->SetType("scale");
        y_grad_op->SetInput("X", OutputGrad(framework::GradVarName("X")));
        y_grad_op->SetOutput("Out", y_grads);
        y_grad_op->SetAttr("scale", 0.0f);
        ops.emplace_back(y_grad_op);
      }
    }

    auto out_grads = InputGrad(framework::GradVarName("Out"));
    if(!out_grads.empty()){
      auto* out_grad_op = new framework::OpDesc();
      if(!ddx.empty() && !ddy.empty()){
        out_grad_op->SetType("elementwise_add");
        out_grad_op->SetInput("X", OutputGrad(framework::GradVarName("X")));
        out_grad_op->SetInput("Y", OutputGrad(framework::GradVarName("Y")));
        out_grad_op->SetAttrMap(Attrs());
        out_grad_op->SetOutput("Out", out_grads);
        ops.emplace_back(out_grad_op);
      } else {
        if(!ddx.empty()){
          out_grad_op->SetType("scale");
          out_grad_op->SetInput("X", ddx);
          out_grad_op->SetOutput("Out", out_grads);
          out_grad_op->SetAttr("scale", 1.0f);
          ops.emplace_back(out_grad_op);
        }
        if(!ddy.empty()){
          out_grad_op->SetType("scale");
          out_grad_op->SetInput("X", ddy);
          out_grad_op->SetOutput("Out", out_grads);
          out_grad_op->SetAttr("scale", 1.0f);
          ops.emplace_back(out_grad_op);
          }
      }
    return ops;
  }
};

} // operators
} //paddle

namespace ops = paddle::operators;
REGISTER_ELEMWISE_GRAD_MAKER(elementwise_add, Add);
//REGISTER_ELEMWISE_EXPLICIT_OP(elementwise_add, "Add", "Out = X + Y");
class __ElemwiseOpelementwise_addMaker__
  : public ::paddle::operators::ElementwiseOpMaker{
 protected:
  virtual std::string GetName() const { return "Add"; }
  virtual std::string GetEquation() const { return "Out = X + Y"; }
};

REGISTER_OPERATOR(elementwise_add, ops::ElementwiseOp,
                  __ElemwiseOpelementwise_addMaker__,
                  ops::ElementwiseOpInferVarType,
                  elementwise_addGradMaker);
REGISTER_OPERATOR(elementwise_add_grad, ops::ElementwiseOpGrad, ops::ElementwiseAddGradGradMaker);


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
