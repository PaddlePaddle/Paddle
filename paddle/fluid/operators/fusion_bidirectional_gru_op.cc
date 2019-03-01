/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#include <string>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/blas.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class FusionBidirectionalGRUOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(%s) of GRUOp should not be null.",
                   "Input");
    PADDLE_ENFORCE(ctx->HasInput("Weight0X"),
                   "Input(%s) of GRUOp should not be null.", "Weight0X");
    PADDLE_ENFORCE(ctx->HasInput("Weight1X"),
                   "Input(%s) of GRUOp should not be null.", "Weight1X");
    PADDLE_ENFORCE(ctx->HasInput("Weight2X"),
                   "Input(%s) of GRUOp should not be null.", "Weight2X");
    PADDLE_ENFORCE(ctx->HasInput("Weight3X"),
                   "Input(%s) of GRUOp should not be null.", "Weight3X");
    PADDLE_ENFORCE(ctx->HasInput("Weight0H"),
                   "Input(%s) of GRUOp should not be null.", "Weight0H");
    PADDLE_ENFORCE(ctx->HasInput("Weight1H"),
                   "Input(%s) of GRUOp should not be null.", "Weight1H");
    PADDLE_ENFORCE(ctx->HasInput("Bias0H"),
                   "Input(%s) of GRUOp should not be null.", "Bias0H");
    PADDLE_ENFORCE(ctx->HasInput("Bias1H"),
                   "Input(%s) of GRUOp should not be null.", "Bias1H");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(%s) of GRUOp should not be null.", "Out");
    auto input_dims = ctx->GetInputDim("X");
    auto weight_x_dims = ctx->GetInputDim("Weight0X");
    auto weight_h_dims = ctx->GetInputDim("Weight0H");
    int input_size = weight_x_dims[1];
    int frame_size = weight_h_dims[0];
    PADDLE_ENFORCE_EQ(input_size, frame_size * 3,
                      "The input_size must be 3 times of frame_size in GRUOp.");
    if (ctx->HasInput("H0")) {
      auto h0_dims = ctx->GetInputDim("H0");
      PADDLE_ENFORCE_EQ(h0_dims[1], frame_size,
                        "The width of H0 must be equal to frame_size.");
    }
    framework::DDim premul_dims = {input_dims[0], input_size};

    auto sufw_x_dims = ctx->GetInputDim("Weight2X");
    framework::DDim sufmul_dims = {input_dims[0], sufw_x_dims[1]};

    ctx->SetOutputDim("mul_out0", premul_dims);
    ctx->SetOutputDim("mul_out1", premul_dims);

    ctx->SetOutputDim("gru_out0", {input_dims[0], frame_size});
    ctx->SetOutputDim("gru_out1", {input_dims[0], frame_size});
    ctx->SetOutputDim("Out", sufmul_dims);
    ctx->SetOutputDim("gate0", {frame_size * 3});
    ctx->SetOutputDim("gate1", {frame_size * 3});
    ctx->ShareLoD("X", "Out");
    ctx->ShareLoD("X", "mul_out0");
    ctx->ShareLoD("X", "mul_out1");
    ctx->ShareLoD("X", "gru_out0");
    ctx->ShareLoD("X", "gru_out1");
  }
};

class FusionBidirectionalGRUOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(LoDTensor) The first input is a LodTensor, which supports "
             "variable-time length input sequence. The underlying tensor in "
             "this LoDTenosr is a matrix with shape (T X 3D), where, T is the "
             "total time steps in this mini-batch, D is the hidden size.");
    AddInput("InitH0",
             "(Tensor, optional) The initial hidden state is an optional "
             "input. This is a tensor with shape (N x D), where N is the "
             "batch size, D is the hidden size.")
        .AsDispensable();
    AddInput("InitH1",
             "(Tensor, optional) The initial hidden state is an optional "
             "input. This is a tensor with shape (N x D), where N is the "
             "batch size, D is the hidden size.")
        .AsDispensable();
    AddInput("Weight0X",
             "(Tensor) The learnable hidden-hidden weight matrix with shape "
             "(D x 3D), where D is the hidden size.");
    AddInput("Weight1X",
             "(Tensor) The learnable hidden-hidden weight matrix with shape "
             "(D x 3D), where D is the hidden size.");
    AddInput("Weight2X",
             "(Tensor) The learnable hidden-hidden weight matrix with shape "
             "(D x 3D), where D is the hidden size.");
    AddInput("Weight3X",
             "(Tensor) The learnable hidden-hidden weight matrix with shape "
             "(D x 3D), where D is the hidden size.");
    AddInput(
        "Weight0H",
        "(Tensor) The learnable hidden-hidden weight matrix with shape "
        "(D x 3D), where D is the hidden size. The elements continuous in "
        "memory can be divided into two parts. The first part are weights of "
        "the update gate and reset gate with shape (D x 2D), and the second "
        "part are weights of output candidate with shape (D x D).");
    AddInput(
        "Weight1H",
        "(Tensor) The learnable hidden-hidden weight matrix with shape "
        "(D x 3D), where D is the hidden size. The elements continuous in "
        "memory can be divided into two parts. The first part are weights of "
        "the update gate and reset gate with shape (D x 2D), and the second "
        "part are weights of output candidate with shape (D x D).");
    AddInput("Bias0X",
             "(Tensor, optional) Bias vector with shape (1 x 3D) concating "
             "bias of the update gate, reset gate and output candidate.");
    AddInput("Bias1X",
             "(Tensor, optional) Bias vector with shape (1 x 3D) concating "
             "bias of the update gate, reset gate and output candidate.");
    AddInput("Bias0H",
             "(Tensor, optional) Bias vector with shape (1 x 3D) concating "
             "bias of the update gate, reset gate and output candidate.");
    AddInput("Bias1H",
             "(Tensor, optional) Bias vector with shape (1 x 3D) concating "
             "bias of the update gate, reset gate and output candidate.");
    AddOutput(
        "Out",
        "(LoDTensor) the hidden state LoDTensor organized in sequences. "
        "This LoDTensor is a matrix with shape (T X D) and has the same LoD "
        "with `BatchGate`.");
    AddOutput(
        "mul_out0",
        "(LoDTensor) the hidden state LoDTensor organized in sequences. "
        "This LoDTensor is a matrix with shape (T X D) and has the same LoD "
        "with `BatchGate`.");
    AddOutput(
        "mul_out1",
        "(LoDTensor) the hidden state LoDTensor organized in sequences. "
        "This LoDTensor is a matrix with shape (T X D) and has the same LoD "
        "with `BatchGate`.");
    AddOutput(
        "gru_out0",
        "(LoDTensor) the hidden state LoDTensor organized in sequences. "
        "This LoDTensor is a matrix with shape (T X D) and has the same LoD "
        "with `BatchGate`.");
    AddOutput(
        "gru_out1",
        "(LoDTensor) the hidden state LoDTensor organized in sequences. "
        "This LoDTensor is a matrix with shape (T X D) and has the same LoD "
        "with `BatchGate`.");
    AddOutput(
        "gate0",
        "(LoDTensor) the hidden state LoDTensor organized in sequences. "
        "This LoDTensor is a matrix with shape (T X D) and has the same LoD "
        "with `BatchGate`.");
    AddOutput(
        "gate1",
        "(LoDTensor) the hidden state LoDTensor organized in sequences. "
        "This LoDTensor is a matrix with shape (T X D) and has the same LoD "
        "with `BatchGate`.");
    AddAttr<std::string>("activation",
                         "(string, default tanh) "
                         "The activation type used for output candidate {h}_t.")
        .SetDefault("tanh");
    AddAttr<std::string>(
        "gate_activation",
        "(string, default sigmoid) "
        "The activation type used in update gate and reset gate.")
        .SetDefault("sigmoid");
    AddAttr<int>("reverse",
                 "(int, defalut: 0) "
                 "which one to reverse.")
        .SetDefault(0);
    AddComment(R"DOC(
GRU Operator implements part calculations of the complete GRU as following:

$$
update\_gate: u_t = actGate(xu_t + W_u * h_{t-1} + b_u) \\
reset\_gate: r_t = actGate(xr_t + W_r * h_{t-1} + b_r)  \\
output\_candidate: {h}_t = actNode(xc_t + W_c * dot(r_t, h_{t-1}) + b_c) \\
output: h_t = dot((1 - u_t), h_{t-1}) + dot(u_t, {h}_t)
$$

@note To implement the complete GRU, fully-connected operator must be used
before to feed xu, xr and xc as the Input of GRU operator.
)DOC");
  }
};

template <typename T>
class NotImpleKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_THROW(
        "CPU is not support for this kernel now. Will be add in the future");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fusion_bidirectional_gru, ops::FusionBidirectionalGRUOp,
                  ops::FusionBidirectionalGRUOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);

REGISTER_OP_CPU_KERNEL(fusion_bidirectional_gru, ops::NotImpleKernel<float>);
