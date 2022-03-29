// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/dot_op.h"

#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/phi/core/infermeta_utils.h"
#include "paddle/phi/infermeta/binary.h"

namespace paddle {
namespace operators {

class DotOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"), ctx.GetPlace());
  }
};

class DotOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() final {
    AddInput("X", "(Tensor) The first input tensor. ");
    AddInput("Y", "(Tensor) The second input tensor. ");
    AddOutput("Out", "(Tensor) The result tensor.");
    AddComment("");
  }
};

class DotGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(
        true, ctx->HasInput("X"),
        platform::errors::PreconditionNotMet("Input(X) should not be null."));
    PADDLE_ENFORCE_EQ(
        true, ctx->HasInput("Y"),
        platform::errors::PreconditionNotMet("Input(Y) should not be null."));
    PADDLE_ENFORCE_EQ(true, ctx->HasInput(framework::GradVarName("Out")),
                      platform::errors::PreconditionNotMet(
                          "Input(Out@GRAD) should not be null."));

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->ShareDim("X", /*->*/ x_grad_name);
      ctx->ShareLoD("X", /*->*/ x_grad_name);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->ShareDim("Y", /*->*/ y_grad_name);
      ctx->ShareLoD("Y", /*->*/ y_grad_name);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.GetPlace());
  }
};

template <typename T>
class DotOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("dot_grad");

    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetAttrMap(this->Attrs());
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

DECLARE_INFER_SHAPE_FUNCTOR(dot, DotInferShapeFunctor,
                            PD_INFER_META(phi::DotInferMeta));

REGISTER_OPERATOR(dot, ops::DotOp, ops::DotOpMaker,
                  ops::DotOpGradMaker<paddle::framework::OpDesc>,
                  ops::DotOpGradMaker<paddle::imperative::OpBase>,
                  DotInferShapeFunctor);

REGISTER_OPERATOR(dot_grad, ops::DotGradOp);

REGISTER_OP_CPU_KERNEL(
    dot, ops::DotKernel<paddle::platform::CPUDeviceContext, float>,
    ops::DotKernel<paddle::platform::CPUDeviceContext, double>,
    ops::DotKernel<paddle::platform::CPUDeviceContext, int>,
    ops::DotKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::DotKernel<paddle::platform::CPUDeviceContext,
                   paddle::platform::complex<float>>,
    ops::DotKernel<paddle::platform::CPUDeviceContext,
                   paddle::platform::complex<double>>);
REGISTER_OP_CPU_KERNEL(
    dot_grad, ops::DotGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::DotGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::DotGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::DotGradKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::DotGradKernel<paddle::platform::CPUDeviceContext,
                       paddle::platform::complex<float>>,
    ops::DotGradKernel<paddle::platform::CPUDeviceContext,
                       paddle::platform::complex<double>>);
