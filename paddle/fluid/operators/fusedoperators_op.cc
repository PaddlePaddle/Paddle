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

#include "paddle/fluid/operators/fusedoperators_op.h"

namespace paddle {
namespace operators {

void FusedOperatorsOp::InferShape(framework::InferShapeContext *ctx) const {
  PADDLE_ENFORCE(ctx->HasInput("X"),
                 "Input(X) of FusedOperatorsOp op should not be null.");
  PADDLE_ENFORCE(ctx->HasInput("Y"),
                 "Input(Y) of FusedOperatorsOp op should not be null.");
  PADDLE_ENFORCE(ctx->HasOutput("Out"),
                 "Output(Out) of FusedOperatorsOp op should not be null.");

  auto x_dim = ctx->GetInputDim("X");
  auto y_dim = ctx->GetInputDim("Y");
  PADDLE_ENFORCE_GE(x_dim.size(), y_dim.size(),
                    "Rank of first input must >= rank of second input.");

  ctx->SetOutputDim("Out", x_dim);
  ctx->ShareLoD("X", /*->*/ "Out");
}

framework::OpKernelType FusedOperatorsOp::GetExpectedKernelType(
    const framework::ExecutionContext &ctx) const {
  PADDLE_ENFORCE_EQ(ctx.Input<Tensor>("X")->type(),
                    ctx.Input<Tensor>("Y")->type(),
                    "The element's type of input should be the same.");
  auto input_data_type = framework::ToDataType(ctx.Input<Tensor>("X")->type());
  return framework::OpKernelType(input_data_type, ctx.GetPlace());
}

void FusedOperatorsMaker::Make() {
  AddInput("X", "(vector<Tensor>)");
  AddInput("Y", "(vector<Tensor>)");
  AddOutput("Out", "vector<Tensor>");
  AddAttr<int>("axis", "").SetDefault(-1);
  AddAttr<std::vector<std::string>>("functor_list", "");

  AddComment(R"DOC(
FusedOperators Operator.

for example,

add;scale,k
div;relu

)DOC");
}

void FusedOperatorsOpGrad::InferShape(framework::InferShapeContext *ctx) const {
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

framework::OpKernelType FusedOperatorsOpGrad::GetExpectedKernelType(
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
REGISTER_OPERATOR(fusedoperators, ops::FusedOperatorsOp,
                  ops::FusedOperatorsMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(fusedoperators_grad, ops::FusedOperatorsOpGrad);

REGISTER_OP_CPU_KERNEL(
    fusedoperators,
    ops::FusedOperatorsKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FusedOperatorsKernel<paddle::platform::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    fusedoperators_grad,
    ops::FusedOperatorsGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::FusedOperatorsGradKernel<paddle::platform::CPUDeviceContext, double>);
