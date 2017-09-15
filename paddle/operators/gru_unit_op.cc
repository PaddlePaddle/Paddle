/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/operators/gru_unit_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class GRUUnitOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("input"),
                   "Input(%s) of GRUUnitOp should not be null.", "input");
    PADDLE_ENFORCE(ctx->HasInput("hidden_prev"),
                   "Input(%s) of GRUUnitOp should not be null.", "hidden_prev");
    PADDLE_ENFORCE(ctx->HasInput("weight"),
                   "Input(%s) of GRUUnitOp should not be null.", "weight");
    PADDLE_ENFORCE(ctx->HasInput("bias"),
                   "Input(%s) of GRUUnitOp should not be null.", "bias");
    PADDLE_ENFORCE(ctx->HasOutput("gate"),
                   "Output(%s) of GRUUnitOp should not be null.", "gate");
    PADDLE_ENFORCE(ctx->HasOutput("reset_hidden_prev"),
                   "Output(%s) of GRUUnitOp should not be null.",
                   "reset_hidden_prev");
    PADDLE_ENFORCE(ctx->HasOutput("hidden"),
                   "Output(%s) of GRUUnitOp should not be null.", "hidden");
    auto input_dims = ctx->GetInputDim("input");
    auto hidden_prev_dims = ctx->GetInputDim("hidden_prev");
    auto weight_dims = ctx->GetInputDim("weight");
    auto bias_dims = ctx->GetInputDim("bias");
    int batch_size = input_dims[0];
    int input_size = input_dims[1];
    int frame_size = hidden_prev_dims[1];
    int weight_height = weight_dims[0];
    int weight_width = weight_dims[1];
    int bias_height = bias_dims[0];
    int bias_width = bias_dims[1];
    PADDLE_ENFORCE_EQ(
        input_size, frame_size * 3,
        "The innput_size must be 3 times of frame_size in GRUUnitOp.");
    PADDLE_ENFORCE_EQ(
        weight_height, frame_size,
        "The shape of weight matrix must be [frame_size, frame_size * 3].");
    PADDLE_ENFORCE_EQ(
        weight_width, frame_size * 3,
        "The shape of weight matrix must be [frame_size, frame_size * 3].");
    PADDLE_ENFORCE_EQ(bias_height, 1,
                      "The shape of bias must be [1, frame_size * 3].");
    PADDLE_ENFORCE_EQ(bias_width, frame_size * 3,
                      "The shape of bias must be [1, frame_size * 3].");
    ctx->SetOutputDim("gate", {batch_size, frame_size * 3});
    ctx->SetOutputDim("reset_hidden_prev", {batch_size, frame_size});
    ctx->SetOutputDim("hidden", {batch_size, frame_size});
  }
};

class GRUUnitOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  GRUUnitOpMaker(framework::OpProto *proto,
                 framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("input",
             "(Tensor) Matrix with shape [batch_size, frame_size * 3] for the "
             "input.");
    AddInput("hidden_prev",
             "(Tensor) Matrix with shape [batch_size, frame_size] for the "
             "states of previous time step.");
    AddInput("weight",
             "(Tensor) Weight matrix with shape [frame_size, frame_size * 3]. "
             "The elements continuous in memory can be divided into two parts. "
             "The first part are weights of the update gate and reset gate "
             "with shape [frame_size, frame_size * 2], and the second part are "
             "weights of output candidate with shape [frame_size, frame_size]");
    AddInput("bias",
             "(Tensor) Bias vector with shape [1, frame_size * 3] concating "
             "bias of the update gate, reset gate and output candidate.");
    AddOutput("gate",
              "(Tensor) Matrix with shape [batch_size, frame_size * 3] for the "
              "output of update gate, reset gate and output candidate")
        .AsIntermediate();
    AddOutput("reset_hidden_prev",
              "(Tensor) Matrix with shape [batch_size, frame_size] for the "
              "reseted hidden state of previous time step.")
        .AsIntermediate();
    AddOutput("hidden",
              "(Tensor) The GRU hidden state of the current time step "
              "with shape [batch_size, frame_size].");
    AddComment(R"DOC(
GRUUnitOp implements part calculations of the GRU unit as following:

\f[
update \ gate: u_t = actGate(xu_t + W_u * hidden_prev + bias_u) \\
reset \ gate: r_t = actGate(xr_t + W_r * hidden_prev + bias_r)  \\
output \ candidate: {h}_t = actNode(xc_t + W_c * dot(r_t, hidden_prev) + bias_c) \\
output: h_t = dot((1-u_t), {h}_t) + dot(u_t, hidden_prev)
\f]

The rest of GRU unit can be completed by using FCOp's output as the input of GRUUnitOp.
)DOC");
  }
};

class GRUUnitGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContextBase *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("input"),
                   "Input(%s) of GRUUnitGradOp should not be null.", "input");
    PADDLE_ENFORCE(ctx->HasInput("hidden_prev"),
                   "Input(%s) of GRUUnitGradOp should not be null.",
                   "hidden_prev");
    PADDLE_ENFORCE(ctx->HasInput("weight"),
                   "Input(%s) of GRUUnitGradOp should not be null.", "weight");
    PADDLE_ENFORCE(ctx->HasInput("bias"),
                   "Input(%s) of GRUUnitGradOp should not be null.", "bias");
    PADDLE_ENFORCE(ctx->HasInput("gate"),
                   "Input(%s) of GRUUnitGradOp should not be null.", "gate");
    PADDLE_ENFORCE(ctx->HasInput("reset_hidden_prev"),
                   "Input(%s) of GRUUnitGradOp should not be null.",
                   "reset_hidden_prev");
    PADDLE_ENFORCE(ctx->HasInput("hidden"),
                   "Input(%s) of GRUUnitGradOp should not be null.", "hidden");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("gate")),
                   "Input(%s@GRAD) of GRUUnitGradOp should not be null.",
                   "gate");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("reset_hidden_prev")),
                   "Input(%s@GRAD) of GRUUnitGradOp should not be null.",
                   "reset_hidden_prev");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("hidden")),
                   "Input(%s@GRAD) of GRUUnitGradOp should not be null.",
                   "hidden");
    auto input_dims = ctx->GetInputDim("input");
    auto hidden_prev_dims = ctx->GetInputDim("hidden_prev");
    auto weight_dims = ctx->GetInputDim("weight");
    auto bias_dims = ctx->GetInputDim("bias");
    // int batch_size = input_dims[0];
    int input_size = input_dims[1];
    int frame_size = hidden_prev_dims[1];
    int weight_height = weight_dims[0];
    int weight_width = weight_dims[1];
    int bias_height = bias_dims[0];
    int bias_width = bias_dims[1];
    PADDLE_ENFORCE_EQ(
        input_size, frame_size * 3,
        "The innput_size must be 3 times of frame_size in GRUUnitOp.");
    PADDLE_ENFORCE_EQ(
        weight_height, frame_size,
        "The shape of weight matrix must be [frame_size, frame_size * 3].");
    PADDLE_ENFORCE_EQ(
        weight_width, frame_size * 3,
        "The shape of weight matrix must be [frame_size, frame_size * 3].");
    PADDLE_ENFORCE_EQ(bias_height, 1,
                      "The shape of bias must be [1, frame_size * 3].");
    PADDLE_ENFORCE_EQ(bias_width, frame_size * 3,
                      "The shape of bias must be [1, frame_size * 3].");
    auto input_grad_name = framework::GradVarName("input");
    if (ctx->HasOutput(input_grad_name))
      ctx->SetOutputDim(input_grad_name, input_dims);
    auto hidden_prev_grad_name = framework::GradVarName("hidden_prev");
    if (ctx->HasOutput(hidden_prev_grad_name))
      ctx->SetOutputDim(hidden_prev_grad_name, hidden_prev_dims);
    auto weight_grad_name = framework::GradVarName("weight");
    if (ctx->HasOutput(weight_grad_name))
      ctx->SetOutputDim(weight_grad_name, weight_dims);
    auto bias_grad_name = framework::GradVarName("bias");
    if (ctx->HasOutput(bias_grad_name))
      ctx->SetOutputDim(bias_grad_name, bias_dims);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(gru_unit, ops::GRUUnitOp, ops::GRUUnitOpMaker, gru_unit_grad,
            ops::GRUUnitGradOp);
REGISTER_OP_CPU_KERNEL(gru_unit,
                       ops::GRUUnitKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    gru_unit_grad, ops::GRUUnitGradKernel<paddle::platform::CPUPlace, float>);
