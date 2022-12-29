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

#include <memory>

namespace paddle {
namespace operators {

class GRUUnitOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "GRUUnit");
    OP_INOUT_CHECK(
        ctx->HasInput("HiddenPrev"), "Input", "HiddenPrev", "GRUUnit");
    OP_INOUT_CHECK(ctx->HasInput("Weight"), "Input", "Weight", "GRUUnit");
    OP_INOUT_CHECK(ctx->HasOutput("Gate"), "Output", "Gate", "GRUUnit");
    OP_INOUT_CHECK(ctx->HasOutput("ResetHiddenPrev"),
                   "Output",
                   "ResetHiddenPrev",
                   "GRUUnit");
    OP_INOUT_CHECK(ctx->HasOutput("Hidden"), "Output", "Hidden", "GRUUnit");
    auto input_dims = ctx->GetInputDim("Input");
    auto hidden_prev_dims = ctx->GetInputDim("HiddenPrev");
    auto weight_dims = ctx->GetInputDim("Weight");
    int batch_size = input_dims[0];
    int input_size = input_dims[1];
    int frame_size = hidden_prev_dims[1];
    int weight_height = weight_dims[0];
    int weight_width = weight_dims[1];
    if (ctx->IsRuntime() || input_size >= 0) {
      PADDLE_ENFORCE_EQ(input_size,
                        frame_size * 3,
                        platform::errors::InvalidArgument(
                            "The second dimension of Input(Input) must be 3 "
                            "times of frame_size in GRUUnitOp, but received %d "
                            "(Input) vs %d (frame_size).",
                            input_size,
                            frame_size));
    }
    PADDLE_ENFORCE_EQ(
        weight_height,
        frame_size,
        platform::errors::InvalidArgument(
            "The shape of Input(Weight) matrix must be [frame_size, frame_size "
            "* 3] in GRUUnitOp, but received [%d, %d] (Weight) vs [%d, %d] "
            "(frame_size).",
            weight_height,
            weight_width,
            frame_size,
            frame_size * 3));
    PADDLE_ENFORCE_EQ(
        weight_width,
        frame_size * 3,
        platform::errors::InvalidArgument(
            "The shape of Input(Weight) matrix must be [frame_size, frame_size "
            "* 3] in GRUUnitOp, but received [%d, %d] (Weight) vs [%d, %d] "
            "(frame_size).",
            weight_height,
            weight_width,
            frame_size,
            frame_size * 3));

    if (ctx->HasInput("Bias")) {
      auto bias_dims = ctx->GetInputDim("Bias");
      int bias_height = bias_dims[0];
      int bias_width = bias_dims[1];
      PADDLE_ENFORCE_EQ(
          bias_height,
          1,
          platform::errors::InvalidArgument(
              "The shape of Bias must be [1, frame_size * 3], but received "
              "[%d, %d] (Bias) vs [1, %d] (frame_size * 3).",
              bias_height,
              bias_width,
              frame_size * 3));
      PADDLE_ENFORCE_EQ(
          bias_width,
          frame_size * 3,
          platform::errors::InvalidArgument(
              "The shape of Bias must be [1, frame_size * 3], but received "
              "[%d, %d] (Bias) vs [1, %d] (frame_size * 3).",
              bias_height,
              bias_width,
              frame_size * 3));
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
              "reset hidden state of previous time step.")
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
    AddAttr<bool>("origin_mode",
                  "bool"
                  "use origin mode in article <Learning Phrase Representations "
                  "using RNN Encoder–Decoder\n"
                  "for Statistical Machine "
                  "Translation>(https://arxiv.org/pdf/1406.1078.pdf)")
        .SetDefault(false);
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
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "GRUUnitGrad");
    OP_INOUT_CHECK(
        ctx->HasInput("HiddenPrev"), "Input", "HiddenPrev", "GRUUnitGrad");
    OP_INOUT_CHECK(ctx->HasInput("Weight"), "Input", "Weight", "GRUUnitGrad");
    OP_INOUT_CHECK(ctx->HasInput("Gate"), "Input", "Gate", "GRUUnitGrad");
    OP_INOUT_CHECK(ctx->HasInput("ResetHiddenPrev"),
                   "Input",
                   "ResetHiddenPrev",
                   "GRUUnitGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Hidden")),
                   "Input",
                   "Hidden@GRAD",
                   "GRUUnitGrad");

    auto input_dims = ctx->GetInputDim("Input");
    auto hidden_prev_dims = ctx->GetInputDim("HiddenPrev");
    auto weight_dims = ctx->GetInputDim("Weight");
    // int batch_size = input_dims[0];
    int input_size = input_dims[1];
    int frame_size = hidden_prev_dims[1];
    int weight_height = weight_dims[0];
    int weight_width = weight_dims[1];
    if (ctx->IsRuntime() || input_size >= 0) {
      PADDLE_ENFORCE_EQ(
          input_size,
          frame_size * 3,
          platform::errors::InvalidArgument(
              "The second dimension of Input(Input) must be 3 "
              "times of frame_size in GRUUnitGradOp, but received %d "
              "(Input) vs %d (frame_size).",
              input_size,
              frame_size));
    }
    PADDLE_ENFORCE_EQ(
        weight_height,
        frame_size,
        platform::errors::InvalidArgument(
            "The shape of Input(Weight) matrix must be [frame_size, frame_size "
            "* 3] in GRUUnitGradOp, but received [%d, %d] (Weight) vs [%d, %d] "
            "(frame_size).",
            weight_height,
            weight_width,
            frame_size,
            frame_size * 3));
    PADDLE_ENFORCE_EQ(
        weight_width,
        frame_size * 3,
        platform::errors::InvalidArgument(
            "The shape of Input(Weight) matrix must be [frame_size, frame_size "
            "* 3] in GRUUnitGradOp, but received [%d, %d] (Weight) vs [%d, %d] "
            "(frame_size).",
            weight_height,
            weight_width,
            frame_size,
            frame_size * 3));
    if (ctx->HasInput("Bias")) {
      auto bias_dims = ctx->GetInputDim("Bias");
      int bias_height = bias_dims[0];
      int bias_width = bias_dims[1];

      PADDLE_ENFORCE_EQ(
          bias_height,
          1,
          platform::errors::InvalidArgument(
              "The shape of Bias must be [1, frame_size * 3], but received "
              "[%d, %d] (Bias) vs [1, %d] (frame_size * 3).",
              bias_height,
              bias_width,
              frame_size * 3));
      PADDLE_ENFORCE_EQ(
          bias_width,
          frame_size * 3,
          platform::errors::InvalidArgument(
              "The shape of Bias must be [1, frame_size * 3], but received "
              "[%d, %d] (Bias) vs [1, %d] (frame_size * 3).",
              bias_height,
              bias_width,
              frame_size * 3));
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

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Hidden")),
                                   ctx.device_context());
  }
};

template <typename T>
class GRUUnitGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("gru_unit_grad");

    op->SetInput("Input", this->Input("Input"));
    op->SetInput("HiddenPrev", this->Input("HiddenPrev"));
    op->SetInput("Weight", this->Input("Weight"));
    op->SetInput("Bias", this->Input("Bias"));

    op->SetInput("Gate", this->Output("Gate"));
    op->SetInput("ResetHiddenPrev", this->Output("ResetHiddenPrev"));
    op->SetInput(framework::GradVarName("Hidden"), this->OutputGrad("Hidden"));

    op->SetAttrMap(this->Attrs());

    op->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
    op->SetOutput(framework::GradVarName("HiddenPrev"),
                  this->InputGrad("HiddenPrev"));
    op->SetOutput(framework::GradVarName("Weight"), this->InputGrad("Weight"));
    op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(GRUUnitGradOpNoNeedBufferVarInferer,
                                    "Bias");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(gru_unit,
                  ops::GRUUnitOp,
                  ops::GRUUnitOpMaker,
                  ops::GRUUnitGradOpMaker<paddle::framework::OpDesc>,
                  ops::GRUUnitGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(gru_unit_grad,
                  ops::GRUUnitGradOp,
                  ops::GRUUnitGradOpNoNeedBufferVarInferer);

REGISTER_OP_CPU_KERNEL(gru_unit,
                       ops::GRUUnitKernel<phi::CPUContext, float>,
                       ops::GRUUnitKernel<phi::CPUContext, double>);
REGISTER_OP_CPU_KERNEL(gru_unit_grad,
                       ops::GRUUnitGradKernel<phi::CPUContext, float>,
                       ops::GRUUnitGradKernel<phi::CPUContext, double>);
