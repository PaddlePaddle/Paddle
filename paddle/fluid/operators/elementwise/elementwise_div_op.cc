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

template <typename T>
struct SameDimsElemwiseDiv<
    platform::CPUDeviceContext, T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  void operator()(const framework::ExecutionContext &ctx,
                  const framework::Tensor *x, const framework::Tensor *y,
                  framework::Tensor *z) {
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(ctx);
    blas.VDIV(x->numel(), x->data<T>(), y->data<T>(), z->data<T>());
  }
};

// use default div function for int32/int64 type because of divison zero
// checking.
template <typename T>
struct SameDimsElemwiseDiv<
    platform::CPUDeviceContext, T,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
  void operator()(const framework::ExecutionContext &ctx,
                  const framework::Tensor *x, const framework::Tensor *y,
                  framework::Tensor *z) {
    default_elementwise_div<platform::CPUDeviceContext, T>(ctx, x, y, z);
  }
};

class ElementwiseDivOpMaker : public ElementwiseOpMaker {
 protected:
  std::string GetName() const override { return "Div"; }
  std::string GetEquation() const override { return "Out = X / Y"; }

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
    return "Divide two tensors element-wise";
  }
};

class ElementwiseDivGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("elementwise_div_grad");
    op->SetInput("Y", Input("Y"));
    op->SetInput("Out", Output("Out"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), InputGrad("Y"));
    op->SetAttrMap(Attrs());
    return op;
  }
};

class ElementwiseDivDoubleGradDescMaker
    : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("elementwise_div_grad_grad");
    op->SetInput("Y", Input("Y"));
    op->SetInput("Out", Input("Out"));
    op->SetInput("DDX", OutputGrad(framework::GradVarName("X")));
    op->SetInput("DDY", OutputGrad(framework::GradVarName("Y")));
    op->SetInput("DX", Output(framework::GradVarName("X")));

    op->SetAttrMap(Attrs());

    op->SetOutput(framework::GradVarName("Y"), InputGrad("Y"));
    op->SetOutput("DOut", InputGrad("Out"));
    op->SetOutput("DDOut", InputGrad(framework::GradVarName("Out")));

    return op;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(elementwise_div, ops::ElementwiseOp,
                  ops::ElementwiseDivOpMaker, ops::ElementwiseOpInferVarType,
                  ops::ElementwiseDivGradOpDescMaker);

REGISTER_OPERATOR(elementwise_div_grad, ops::ElementwiseOpGrad,
                  ops::ElementwiseDivDoubleGradDescMaker);
REGISTER_OPERATOR(elementwise_div_grad_grad, ops::ElementwiseDivOpDoubleGrad,
                  ops::ElementwiseDivDoubleGradOpInplace);

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
