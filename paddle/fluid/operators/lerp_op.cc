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

    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
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
)DOC");
  }
};

class LerpGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@Grad", "lerp_grad");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto dtype = framework::OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(dtype, ctx.GetPlace());
  }
};

template <typename T>
class LerpOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> op) const override {
    //
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
