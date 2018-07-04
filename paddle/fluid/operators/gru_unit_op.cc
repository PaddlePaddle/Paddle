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

#include "paddle/fluid/operators/gru_unit_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class GRUUnitOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"),
                   "Input(%s) of GRUUnitOp should not be null.", "Input");
    PADDLE_ENFORCE(ctx->HasInput("HiddenPrev"),
                   "Input(%s) of GRUUnitOp should not be null.", "HiddenPrev");
    PADDLE_ENFORCE(ctx->HasInput("Weight"),
                   "Input(%s) of GRUUnitOp should not be null.", "Weight");
    PADDLE_ENFORCE(ctx->HasOutput("Gate"),
                   "Output(%s) of GRUUnitOp should not be null.", "Gate");
    PADDLE_ENFORCE(ctx->HasOutput("ResetHiddenPrev"),
                   "Output(%s) of GRUUnitOp should not be null.",
                   "ResetHiddenPrev");
    PADDLE_ENFORCE(ctx->HasOutput("Hidden"),
                   "Output(%s) of GRUUnitOp should not be null.", "Hidden");
    auto input_dims = ctx->GetInputDim("Input");
    auto hidden_prev_dims = ctx->GetInputDim("HiddenPrev");
    auto weight_dims = ctx->GetInputDim("Weight");
    int batch_size = input_dims[0];
    int input_size = input_dims[1];
    int frame_size = hidden_prev_dims[1];
    int weight_height = weight_dims[0];
    int weight_width = weight_dims[1];
    PADDLE_ENFORCE_EQ(
        input_size, frame_size * 3,
        "The input_size must be 3 times of frame_size in GRUUnitOp.");
    PADDLE_ENFORCE_EQ(
        weight_height, frame_size,
        "The shape of Weight matrix must be [frame_size, frame_size * 3].");
    PADDLE_ENFORCE_EQ(
        weight_width, frame_size * 3,
        "The shape of Weight matrix must be [frame_size, frame_size * 3].");
    if (ctx->HasInput("Bias")) {
      auto bias_dims = ctx->GetInputDim("Bias");
      int bias_height = bias_dims[0];
      int bias_width = bias_dims[1];
      PADDLE_ENFORCE_EQ(bias_height, 1,
                        "The shape of Bias must be [1, frame_size * 3].");
      PADDLE_ENFORCE_EQ(bias_width, frame_size * 3,
                        "The shape of Bias must be [1, frame_size * 3].");
    }
    ctx->SetOutputDim("Gate", {batch_size, frame_size * 3});
    ctx->SetOutputDim("ResetHiddenPrev", {batch_size, frame_size});
    ctx->SetOutputDim("Hidden", {batch_size, frame_size});
  }
};

class GRUUnitOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "(Tensor) Matrix with shape [batch_size, frame_size * 3] for the "
             "input.");
    AddInput("HiddenPrev",
             "(Tensor) Matrix with shape [batch_size, frame_size] for the "
             "states of previous time step.");
    AddInput(
        "Weight",
        "(Tensor) Weight matrix with shape [frame_size, frame_size * 3]. "
        "The elements continuous in memory can be divided into two parts. "
        "The first part are weights of the update gate and reset gate "
        "with shape [frame_size, frame_size * 2], and the second part are "
        "weights of output candidate with shape [frame_size, frame_size].");
    AddInput(
        "Bias",
        "(Tensor) Bias vector with shape [1, frame_size * 3] concatenating "
        "bias of the update gate, reset gate and output candidate.")
        .AsDispensable();
    AddOutput("Gate",
              "(Tensor) Matrix with shape [batch_size, frame_size * 3] for the "
              "output of update gate, reset gate and output candidate.")
        .AsIntermediate();
    AddOutput("ResetHiddenPrev",
              "(Tensor) Matrix with shape [batch_size, frame_size] for the "
              "reseted hidden state of previous time step.")
        .AsIntermediate();
    AddOutput("Hidden",
              "(Tensor) The GRU hidden state of the current time step "
              "with shape [batch_size, frame_size].");
    AddAttr<int>("activation",
                 "(enum int, default tanh) "
                 "The activation type used for output candidate {h}_t.")
        .SetDefault(tanh)
        .InEnum({identity, sigmoid, tanh, relu});
    AddAttr<int>("gate_activation",
                 "(enum int, default sigmoid) "
                 "The activation type used in update gate and reset gate.")
        .SetDefault(sigmoid)
        .InEnum({identity, sigmoid, tanh, relu});
    AddComment(R"DOC(
GRUUnit Operator implements partial calculations of the GRU unit as following:

$$
update \ gate: u_t = actGate(xu_t + W_u * h_{t-1} + b_u) \\
reset \ gate: r_t = actGate(xr_t + W_r * h_{t-1} + b_r)  \\
output \ candidate: {h}_t = actNode(xc_t + W_c * dot(r_t, h_{t-1}) + b_c) \\
output: h_t = dot((1 - u_t), h_{t-1}) + dot(u_t, {h}_t)
$$

which is same as one time step of GRU Operator.

@note To implement the complete GRU unit, fully-connected operator must be
used before to feed xu, xr and xc as the Input of GRUUnit operator.

)DOC");
  }
};

class GRUUnitGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"),
                   "Input(%s) of GRUUnitGradOp should not be null.", "Input");
    PADDLE_ENFORCE(ctx->HasInput("HiddenPrev"),
                   "Input(%s) of GRUUnitGradOp should not be null.",
                   "HiddenPrev");
    PADDLE_ENFORCE(ctx->HasInput("Weight"),
                   "Input(%s) of GRUUnitGradOp should not be null.", "Weight");
    PADDLE_ENFORCE(ctx->HasInput("Gate"),
                   "Input(%s) of GRUUnitGradOp should not be null.", "Gate");
    PADDLE_ENFORCE(ctx->HasInput("ResetHiddenPrev"),
                   "Input(%s) of GRUUnitGradOp should not be null.",
                   "ResetHiddenPrev");
    PADDLE_ENFORCE(ctx->HasInput("Hidden"),
                   "Input(%s) of GRUUnitGradOp should not be null.", "Hidden");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Hidden")),
                   "Input(%s@GRAD) of GRUUnitGradOp should not be null.",
                   "Hidden");
    auto input_dims = ctx->GetInputDim("Input");
    auto hidden_prev_dims = ctx->GetInputDim("HiddenPrev");
    auto weight_dims = ctx->GetInputDim("Weight");
    // int batch_size = input_dims[0];
    int input_size = input_dims[1];
    int frame_size = hidden_prev_dims[1];
    int weight_height = weight_dims[0];
    int weight_width = weight_dims[1];
    PADDLE_ENFORCE_EQ(
        input_size, frame_size * 3,
        "The input_size must be 3 times of frame_size in GRUUnitOp.");
    PADDLE_ENFORCE_EQ(
        weight_height, frame_size,
        "The shape of Weight matrix must be [frame_size, frame_size * 3].");
    PADDLE_ENFORCE_EQ(
        weight_width, frame_size * 3,
        "The shape of Weight matrix must be [frame_size, frame_size * 3].");
    if (ctx->HasInput("Bias")) {
      auto bias_dims = ctx->GetInputDim("Bias");
      int bias_height = bias_dims[0];
      int bias_width = bias_dims[1];
      PADDLE_ENFORCE_EQ(bias_height, 1,
                        "The shape of Bias must be [1, frame_size * 3].");
      PADDLE_ENFORCE_EQ(bias_width, frame_size * 3,
                        "The shape of Bias must be [1, frame_size * 3].");
      auto bias_grad_name = framework::GradVarName("Bias");
      if (ctx->HasOutput(bias_grad_name))
        ctx->SetOutputDim(bias_grad_name, bias_dims);
    }
    auto input_grad_name = framework::GradVarName("Input");
    if (ctx->HasOutput(input_grad_name))
      ctx->SetOutputDim(input_grad_name, input_dims);
    auto hidden_prev_grad_name = framework::GradVarName("HiddenPrev");
    if (ctx->HasOutput(hidden_prev_grad_name))
      ctx->SetOutputDim(hidden_prev_grad_name, hidden_prev_dims);
    auto weight_grad_name = framework::GradVarName("Weight");
    if (ctx->HasOutput(weight_grad_name))
      ctx->SetOutputDim(weight_grad_name, weight_dims);
  }
};

class GRUUnitGradOpMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto* op = new framework::OpDesc();
    op->SetType("gru_unit_grad");

    op->SetInput("Input", Input("Input"));
    op->SetInput("HiddenPrev", Input("HiddenPrev"));
    op->SetInput("Weight", Input("Weight"));
    op->SetInput("Bias", Input("Bias"));

    op->SetInput("Hidden", Output("Hidden"));
    op->SetInput("Gate", Output("Gate"));
    op->SetInput("ResetHiddenPrev", Output("ResetHiddenPrev"));
    op->SetInput(framework::GradVarName("Hidden"), OutputGrad("Hidden"));

    op->SetAttrMap(Attrs());

    op->SetOutput(framework::GradVarName("Input"), InputGrad("Input"));
    op->SetOutput(framework::GradVarName("HiddenPrev"),
                  InputGrad("HiddenPrev"));
    op->SetOutput(framework::GradVarName("Weight"), InputGrad("Weight"));
    op->SetOutput(framework::GradVarName("Bias"), InputGrad("Bias"));
    return std::unique_ptr<framework::OpDesc>(op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(gru_unit, ops::GRUUnitOp, ops::GRUUnitOpMaker,
                  ops::GRUUnitGradOpMaker);
REGISTER_OPERATOR(gru_unit_grad, ops::GRUUnitGradOp);

REGISTER_OP_CPU_KERNEL(
    gru_unit, ops::GRUUnitKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GRUUnitKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    gru_unit_grad,
    ops::GRUUnitGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::GRUUnitGradKernel<paddle::platform::CPUDeviceContext, double>);
