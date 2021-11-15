// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/lerp_op.h"

namespace paddle {
namespace operators {

class LerpOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "lerp");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "lerp");
    OP_INOUT_CHECK(ctx->HasInput("Weight"), "Input", "Weight", "lerp");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "lerp");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");
    auto w_dims = ctx->GetInputDim("Weight");
    VLOG(3) << "lerp x.shape = " << x_dims << ", y.shape = " << y_dims
            << ", weight.shape = " << w_dims;

    ctx->SetOutputDim("Out", x_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class LerpOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of lerp op.");
    AddInput("Y", "(Tensor), The input tensor of lerp op.");
    AddInput("Weight", "(float|Tensor), The input tensor of lerp op.");
    AddOutput("Out", "(Tensor), The output tensor of lerp op.");
    AddComment(R"DOC(
Lerp Operator.

This operator is used to do a linear interpolation of input $X$ and $Y$ with $Weight$.

The equation is:

$$Out = X + Weight * (Y - X)$$

Both the input $X$ and $Y$ can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD information with input $X$.

)DOC");
  }
};

class LerpGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "lerp");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "lerp");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "lerp");

    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    auto x_grad_name = framework::GradVarName("X");
    auto y_grad_name = framework::GradVarName("Y");

    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, y_dims);
    }
  }
};

template <typename T>
class LerpOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> op) const override {
    op->SetType("mul_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput("Y", this->Input("Y"));
    op->SetInput("Weight", this->Input("Weight"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->OutputGrad("X"));
    op->SetOutput(framework::GradVarName("Y"), this->OutputGrad("Y"));
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(
    lerp, paddle::operators::LerpOp, paddle::operators::LerpOpMaker,
    paddle::operators::LerpOpGradMaker<paddle::framework::OpDesc>,
    paddle::operators::LerpOpGradMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(lerp_grad, paddle::operators::LerpGradOp);

REGISTER_OP_CPU_KERNEL(
    lerp,
    paddle::operators::LerpKernel<paddle::platform::CPUDeviceContext, float>,
    paddle::operators::LerpKernel<paddle::platform::CPUDeviceContext, double>,
    paddle::operators::LerpKernel<paddle::platform::CPUDeviceContext, int>,
    paddle::operators::LerpKernel<paddle::platform::CPUDeviceContext, int64_t>);

REGISTER_OP_CPU_KERNEL(
    lerp_grad,
    paddle::operators::LerpGradKernel<paddle::platform::CPUDeviceContext,
                                      float>,
    paddle::operators::LerpGradKernel<paddle::platform::CPUDeviceContext,
                                      double>,
    paddle::operators::LerpGradKernel<paddle::platform::CPUDeviceContext, int>,
    paddle::operators::LerpGradKernel<paddle::platform::CPUDeviceContext,
                                      int64_t>);
