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

#include <string>
#include <vector>

#include "paddle/fluid/operators/fused_elemwise_activation_op.h"

namespace paddle {
namespace operators {

void FusedElemwiseActivationOp::InferShape(
    framework::InferShapeContext *ctx) const {
  PADDLE_ENFORCE(
      ctx->HasInput("X"),
      "Input(X) of FusedElemwiseActivationOp op should not be null.");
  PADDLE_ENFORCE(
      ctx->HasInput("Y"),
      "Input(Y) of FusedElemwiseActivationOp op should not be null.");
  PADDLE_ENFORCE(
      ctx->HasOutput("Out"),
      "Output(Out) of FusedElemwiseActivationOp op should not be null.");

  auto x_dim = ctx->GetInputDim("X");
  auto y_dim = ctx->GetInputDim("Y");
  PADDLE_ENFORCE_GE(x_dim.size(), y_dim.size(),
                    "Rank of first input must >= rank of second input.");

  ctx->SetOutputDim("Out", x_dim);
  ctx->ShareLoD("X", /*->*/ "Out");
}

framework::OpKernelType FusedElemwiseActivationOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  PADDLE_ENFORCE_EQ(ctx.Input<Tensor>("X")->type(),
                    ctx.Input<Tensor>("Y")->type(),
                    "The element's type of input should be the same.");
  auto input_data_type = framework::ToDataType(ctx.Input<Tensor>("X")->type());
  return framework::OpKernelType(input_data_type, ctx.GetPlace());
}

void FusedElemwiseActivationMaker::Make() {
  AddInput("X", "(vector<Tensor>)");
  AddInput("Y", "(vector<Tensor>)");
  AddOutput("Out", "vector<Tensor>");
  AddAttr<int>("axis",
               "axis is used by elementwise_op, the default value is -1.")
      .SetDefault(-1);
  AddAttr<float>("scale",
                 "scale is used by scale_op, the default value is 0.0.")
      .SetDefault(0.0);

  AddAttr<std::string>("functor_list", "The functors that should be fused.")
      .AddCustomChecker([](const std::string &functor_list) {
        PADDLE_ENFORCE(math::ValidCheck(functor_list));
      });

  AddComment(R"DOC(
FusedElemwiseActivation Operator.

At present, FusedElemwiseActivation supports only two combinations of two
types(elementwise_op and activation_op) of Op.

    z = f1(x, f2(y))
    z = f1(f2(x, y))

for example:
  functor_list(f1, f2) can be represented as 'add,scale' or 'relu,add'.


)DOC");
}

void FusedElemwiseActivationOpGrad::InferShape(
    framework::InferShapeContext *ctx) const {
  PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
  PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) should not be null");
  PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                 "Input(Out@GRAD) should not be null");

  auto x_dims = ctx->GetInputDim("X");
  auto y_dims = ctx->GetInputDim("Y");
  auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));

  PADDLE_ENFORCE_GE(x_dims.size(), y_dims.size(),
                    "Rank of first input must >= rank of second input.");

  auto x_grad_name = framework::GradVarName("X");
  auto y_grad_name = framework::GradVarName("Y");
  if (ctx->HasOutput(x_grad_name)) {
    ctx->SetOutputDim(x_grad_name, x_dims);
  }
  if (ctx->HasOutput(y_grad_name)) {
    ctx->SetOutputDim(y_grad_name, y_dims);
  }
}

framework::OpKernelType FusedElemwiseActivationOpGrad::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  auto input_data_type_index = ctx.Input<Tensor>("X")->type();
  PADDLE_ENFORCE_EQ(input_data_type_index, ctx.Input<Tensor>("Y")->type(),
                    "The element's type of input should be the same.");
  PADDLE_ENFORCE_EQ(input_data_type_index,
                    ctx.Input<Tensor>(framework::GradVarName("Out"))->type(),
                    "The element's type of input should be the same.");

  auto input_data_type = framework::ToDataType(input_data_type_index);
  return framework::OpKernelType(input_data_type, ctx.GetPlace());
}

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fused_elemwise_activation, ops::FusedElemwiseActivationOp,
                  ops::FusedElemwiseActivationMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(fused_elemwise_activation_grad,
                  ops::FusedElemwiseActivationOpGrad);

REGISTER_OP_CPU_KERNEL(
    fused_elemwise_activation,
    ops::FusedElemwiseActivationKernel<paddle::platform::CPUDeviceContext,
                                       float>,
    ops::FusedElemwiseActivationKernel<paddle::platform::CPUDeviceContext,
                                       double>);

REGISTER_OP_CPU_KERNEL(
    fused_elemwise_activation_grad,
    ops::FusedElemwiseActivationGradKernel<paddle::platform::CPUDeviceContext,
                                           float>,
    ops::FusedElemwiseActivationGradKernel<paddle::platform::CPUDeviceContext,
                                           double>);
