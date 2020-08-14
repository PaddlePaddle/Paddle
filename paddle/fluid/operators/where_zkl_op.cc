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

#include "paddle/fluid/operators/where_zkl_op.h"

namespace paddle {
namespace operators {

class WhereZklOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Condition"), "Input", "Condition",
                   "where_zkl");
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "where_zkl");
    OP_INOUT_CHECK(ctx->HasInput("Y"), "Input", "Y", "where_zkl");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "where_zkl");

    auto condition_dims = ctx->GetInputDim("Condition");
    auto x_dims = ctx->GetInputDim("X");
    auto y_dims = ctx->GetInputDim("Y");

    VLOG(3) << "where_zkl operator condition.shape=" << condition_dims
            << " x.shape=" << x_dims << " y.shape=" << y_dims;

    // PADDLE_ENFORCE_EQ(
    //     x_dims.size(), 1,
    //     platform::errors::InvalidArgument("X should be 1-D."));
    PADDLE_ENFORCE_EQ(condition_dims.size(), x_dims.size(),
                      platform::errors::InvalidArgument(
                          "Condition and X should have the same shape size."));
    PADDLE_ENFORCE_EQ(x_dims.size(), y_dims.size(),
                      platform::errors::InvalidArgument(
                          "X and Y should have the same shape size."));

    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", /*->*/ "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class WhereZklOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Condition",
             "(Tensor) A bool tensor of where_zkl operator."
             "When Condition is true, the value of out is X, otherwise is Y");
    AddInput("X",
             "(Tensor) The input tensor of where_zkl operator."
             "When Condition is true, the value of out is X, otherwise is Y");
    AddInput("Y",
             "(Tensor) The input tensor of where_zkl operator."
             "When Condition is true, the value of out is X, otherwise is Y");
    AddOutput("Out", "(Tensor) The output tensor of where_zkl operator.");
    AddComment(R"DOC(
**Where operator**

According Condition(i) decide Out[i] = X[i] or Y[i].

if condition(i)=True:

$$Out[i] = X[i]$$

else:

$$Out[i] = Y[i]$$
)DOC");
  }
};

class WhereZklGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Condition"), "Input", "Condition",
                   "where_zkl");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   "Out@GRAD", "where_zkl");

    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")), "Output",
                   "X@GRAD", "where_zkl");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("Y")), "Output",
                   "Y@GRAD", "where_zkl");

    ctx->SetOutputDim(framework::GradVarName("X"),
                      ctx->GetInputDim(framework::GradVarName("Out")));
    ctx->SetOutputDim(framework::GradVarName("Y"),
                      ctx->GetInputDim(framework::GradVarName("Out")));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

template <typename T>
class WhereZklOpGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> retv) const override {
    retv->SetType("where_zkl_grad");
    retv->SetInput("Condition", this->Input("Condition"));
    retv->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    retv->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    retv->SetOutput(framework::GradVarName("Y"), this->InputGrad("Y"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(where_zkl, ops::WhereZklOp, ops::WhereZklOpMaker,
                  ops::WhereZklOpGradMaker<paddle::framework::OpDesc>,
                  ops::WhereZklOpGradMaker<paddle::imperative::OpBase>);
// REGISTER_OPERATOR(
//     where_zkl, ops::WhereZklOp, ops::WhereZklOpMaker,
//     paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
//     paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(where_zkl_grad, ops::WhereZklGradOp);

REGISTER_OP_CPU_KERNEL(
    where_zkl, ops::WhereZklKernel<paddle::platform::CPUDeviceContext, float>,
    ops::WhereZklKernel<paddle::platform::CPUDeviceContext, double>,
    ops::WhereZklKernel<paddle::platform::CPUDeviceContext, int>,
    ops::WhereZklKernel<paddle::platform::CPUDeviceContext, int64_t>);

REGISTER_OP_CPU_KERNEL(
    where_zkl_grad,
    ops::WhereZklGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::WhereZklGradKernel<paddle::platform::CPUDeviceContext, double>,
    ops::WhereZklGradKernel<paddle::platform::CPUDeviceContext, int>,
    ops::WhereZklGradKernel<paddle::platform::CPUDeviceContext, int64_t>);
