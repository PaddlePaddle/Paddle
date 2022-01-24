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
struct SameDimsElemwiseAdd<
    platform::CPUDeviceContext, T,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  void operator()(const framework::ExecutionContext &ctx,
                  const framework::Tensor *x, const framework::Tensor *y,
                  framework::Tensor *z) {
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(ctx);
    blas.VADD(x->numel(), x->data<T>(), y->data<T>(), z->data<T>());
  }
};

template <typename T>
struct SameDimsElemwiseAdd<
    platform::CPUDeviceContext, T,
    typename std::enable_if<!std::is_floating_point<T>::value>::type> {
  void operator()(const framework::ExecutionContext &ctx,
                  const framework::Tensor *x, const framework::Tensor *y,
                  framework::Tensor *z) {
    auto eigen_x = framework::EigenVector<T>::Flatten(*x);
    auto eigen_y = framework::EigenVector<T>::Flatten(*y);
    auto eigen_z = framework::EigenVector<T>::Flatten(*z);
    auto &place = *ctx.template device_context<platform::CPUDeviceContext>()
                       .eigen_device();
    eigen_z.device(place) = eigen_x + eigen_y;
  }
};

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
REGISTER_ELEMWISE_EXPLICIT_OP_WITHOUT_GRAD(elementwise_add, Add);

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    elementwise_add_grad, ops::ElementwiseOpGrad,
    ops::ElementwiseGradOpInplaceInferer, ops::ElementwiseGradNoBufVarsInferer,
    ops::ElementwiseAddDoubleGradMaker<paddle::framework::OpDesc>,
    ops::ElementwiseAddDoubleGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(
    elementwise_add_grad_grad, ops::ElementwiseOpDoubleGradWithoutDXDY,
    ops::ElementwiseDoubleGradOpInplaceInferer,
    ops::ElementwiseDoubleGradNoBufVarsInferer,
    ops::ElementwiseAddTripleGradMaker<paddle::framework::OpDesc>,
    ops::ElementwiseAddTripleGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(elementwise_add_triple_grad, ops::ElementwiseOpTripleGrad,
                  ops::ElementwiseTripleGradOpInplaceInferer,
                  ops::ElementwiseTripleGradNoBufVarsInferer);

REGISTER_OP_CPU_KERNEL(
    elementwise_add,
    ops::ElementwiseAddKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwiseAddKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwiseAddKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseAddKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::ElementwiseAddKernel<paddle::platform::CPUDeviceContext,
                              paddle::platform::complex<float>>,
    ops::ElementwiseAddKernel<paddle::platform::CPUDeviceContext,
                              paddle::platform::complex<double>>);
REGISTER_OP_CPU_KERNEL(
    elementwise_add_grad,
    ops::ElementwiseAddGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwiseAddGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwiseAddGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseAddGradKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::ElementwiseAddGradKernel<paddle::platform::CPUDeviceContext,
                                  paddle::platform::complex<float>>,
    ops::ElementwiseAddGradKernel<paddle::platform::CPUDeviceContext,
                                  paddle::platform::complex<double>>);
REGISTER_OP_CPU_KERNEL(
    elementwise_add_grad_grad,
    ops::ElementwiseAddDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        float>,
    ops::ElementwiseAddDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        double>,
    ops::ElementwiseAddDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        int>,
    ops::ElementwiseAddDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        int64_t>,
    ops::ElementwiseAddDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        paddle::platform::complex<float>>,
    ops::ElementwiseAddDoubleGradKernel<paddle::platform::CPUDeviceContext,
                                        paddle::platform::complex<double>>);
REGISTER_OP_CPU_KERNEL(
    elementwise_add_triple_grad,
    ops::ElementwiseAddTripleGradKernel<paddle::platform::CPUDeviceContext,
                                        float>,
    ops::ElementwiseAddTripleGradKernel<paddle::platform::CPUDeviceContext,
                                        double>,
    ops::ElementwiseAddTripleGradKernel<paddle::platform::CPUDeviceContext,
                                        int>,
    ops::ElementwiseAddTripleGradKernel<paddle::platform::CPUDeviceContext,
                                        int64_t>,
    ops::ElementwiseAddTripleGradKernel<paddle::platform::CPUDeviceContext,
                                        paddle::platform::complex<float>>,
    ops::ElementwiseAddTripleGradKernel<paddle::platform::CPUDeviceContext,
                                        paddle::platform::complex<double>>);

// A specialization elementwise_add operator, used in gradient accumulation with
// inplace addto.
REGISTER_OPERATOR(
    grad_add, paddle::operators::ElementwiseOp,
    paddle::operators::ElementwiseAddOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OP_CPU_KERNEL(
    grad_add,
    ops::ElementwiseAddKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ElementwiseAddKernel<paddle::platform::CPUDeviceContext, double>,
    ops::ElementwiseAddKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ElementwiseAddKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::ElementwiseAddKernel<paddle::platform::CPUDeviceContext,
                              paddle::platform::complex<float>>,
    ops::ElementwiseAddKernel<paddle::platform::CPUDeviceContext,
                              paddle::platform::complex<double>>);

REGISTER_OP_VERSION(elementwise_add)
    .AddCheckpoint(
        R"ROC(Register elementwise_add for adding the attribute of
       Scale_y)ROC",
        paddle::framework::compatible::OpVersionDesc().NewAttr(
            "Scale_y",
            "In order to support the function of scaling the input Y when "
            "using the operator of elementwise_add.",
            1.0f));
