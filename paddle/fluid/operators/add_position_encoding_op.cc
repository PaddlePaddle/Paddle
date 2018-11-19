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

#include "paddle/fluid/operators/add_position_encoding_op.h"

namespace paddle {
namespace operators {

class AddPositionEncodingOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "X(Input) of add_position_encoding_op should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("Out"),
        "Out(Output) of add_position_encoding_op should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", x_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class AddPositionEncodingOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "X(Input) must not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Out"), "Out must not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Out@GRAD must not be null.");

    auto out_dims = ctx->GetInputDim("Out");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), out_dims);
    }
  }
};

class AddPositionEncodingOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "Input of AddPositionEncoding operator");
    AddOutput("Out", "Output of AddPositionEncoding operator");
    AddAttr<float>("alpha", "The scale of Original Embedding.")
        .SetDefault(1.0f)
        .AddCustomChecker([](const float& alpha) {
          PADDLE_ENFORCE(alpha >= 0.0f, "'alpha' must be above 0.0.");
        });
    AddAttr<float>("beta", "The scale of Position Embedding.")
        .SetDefault(1.0f)
        .AddCustomChecker([](const float& beta) {
          PADDLE_ENFORCE(beta >= 0.0f, "'beta' must be between 0.0.");
        });
    AddComment(R"DOC(
    Add Position Encoding Operator.
    
    The add position encoding calculates the output based on the input, alpha, beta.
    The size of each dimension of the parameters checked in the infer-shape.
  )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plt = paddle::platform;

REGISTER_OPERATOR(add_position_encoding, ops::AddPositionEncodingOp,
                  ops::AddPositionEncodingOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(add_position_encoding_grad, ops::AddPositionEncodingOpGrad);

REGISTER_OP_CPU_KERNEL(
    add_position_encoding,
    ops::AddPositionEncodingKernel<plt::CPUDeviceContext, float>,
    ops::AddPositionEncodingKernel<plt::CPUDeviceContext, double>);

REGISTER_OP_CPU_KERNEL(
    add_position_encoding_grad,
    ops::AddPositionEncodingGradKernel<plt::CPUDeviceContext, float>,
    ops::AddPositionEncodingGradKernel<plt::CPUDeviceContext, double>);
