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

#include "paddle/fluid/operators/lstm_unit_op.h"

namespace paddle {
namespace operators {

class LstmUnitOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) of LSTM should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("C_prev"),
                   "Input(C_prev) of LSTM should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("C"),
                   "Output(C) of LSTM should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("H"),
                   "Output(H) of LSTM should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    auto c_prev_dims = ctx->GetInputDim("C_prev");

    PADDLE_ENFORCE_EQ(x_dims.size(), 2, "Input(X)'s rank must be 2.");
    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(x_dims[0], c_prev_dims[0],
                        "Batch size of inputs and states must be equal");
      PADDLE_ENFORCE_EQ(x_dims[1], c_prev_dims[1] * 4,
                        "Dimension of FC should equal to prev state * 4");
    }

    int b_size = c_prev_dims[0];  // batch size
    int s_dim = c_prev_dims[1];   // state dim
    ctx->SetOutputDim("C", {b_size, s_dim});
    ctx->SetOutputDim("H", {b_size, s_dim});
  }
};

class LstmUnitOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "Lstm unit only applies non-linear activations, please make sure"
             "that linear tranformation has already been applied to `X`. "
             "Linear tranformation can be applied by adding a `fc` layer");
    AddInput(
        "C_prev",
        "The cell state tensor of last time-step in the Lstm Unit operator.");
    AddOutput("C", "The cell tensor of Lstm Unit operator.");
    AddOutput("H", "The hidden state tensor of Lstm Unit operator.");
    AddAttr<float>("forget_bias",
                   "(float, default 0.0) "
                   "The forget bias of Lstm Unit.")
        .SetDefault(0.0);
    AddComment(R"DOC(
Lstm Unit Operator

Equation:

$$
i, f, o, j = split(X) \\
C = C_{prev} * sigm(f + forget\_bias) + sigm(i) * tanh(j) \\
H = C * sigm(o)
$$

)DOC");
  }
};

class LstmUnitGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("C")),
                   "Input(C@GRAD) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("H")),
                   "Input(H@GRAD) should not be null");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    ctx->SetOutputDim(framework::GradVarName("C_prev"),
                      ctx->GetInputDim("C_prev"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(lstm_unit, ops::LstmUnitOp, ops::LstmUnitOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(lstm_unit_grad, ops::LstmUnitGradOp);
REGISTER_OP_CPU_KERNEL(lstm_unit,
                       ops::LstmUnitKernel<paddle::platform::CPUPlace, float>,
                       ops::LstmUnitKernel<paddle::platform::CPUPlace, double>);
REGISTER_OP_CPU_KERNEL(
    lstm_unit_grad, ops::LstmUnitGradKernel<paddle::platform::CPUPlace, float>,
    ops::LstmUnitGradKernel<paddle::platform::CPUPlace, double>);
