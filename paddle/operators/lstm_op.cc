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

#include "paddle/operators/lstm_op.h"

namespace paddle {
namespace operators {

class LSTMOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"),
                   "Input(Input) of LSTM should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Hidden"),
                   "Output(Hidden) of LSTM should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Cell"),
                   "Output(Cell) of LSTM should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("BatchGate"),
                   "Output(BatchGate) of LSTM should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("BatchCellPreAct"),
                   "Output(BatchGate) of LSTM should not be null.");

    auto in_dims = ctx->GetInputDim("Input");
    PADDLE_ENFORCE_EQ(in_dims.size(), 2, "Input(X)'s rank must be 2.");

    if (ctx->HasInput("H0")) {
      PADDLE_ENFORCE(ctx->HasInput("C0"),
                     "Input(Cell) and Input(Hidden) of LSTM should not "
                     "be null at the same time.");
      auto h_dims = ctx->GetInputDim("H0");
      auto c_dims = ctx->GetInputDim("C0");
      PADDLE_ENFORCE(h_dims == c_dims,
                     "The dimension of Input(H0) and Input(C0) "
                     "should be the same.");
    }

    int frame_size = in_dims[1] / 4;
    auto w_dims = ctx->GetInputDim("Weight");
    PADDLE_ENFORCE_EQ(w_dims.size(), 2,
                      "The rank of Input(Weight) should be 2.");
    PADDLE_ENFORCE_EQ(w_dims[0], frame_size,
                      "The first dimension of Input(Weight) "
                      "should be %d.",
                      frame_size);
    PADDLE_ENFORCE_EQ(w_dims[1], 4 * frame_size,
                      "The second dimension of Input(Weight) "
                      "should be 4 * %d.",
                      frame_size);
    auto b_dims = ctx->GetInputDim("Bias");
    PADDLE_ENFORCE_EQ(b_dims.size(), 2, "The rank of Input(Bias) should be 2.");
    PADDLE_ENFORCE_EQ(b_dims[0], 1,
                      "The first dimension of Input(Bias) should be 1.");
    if (ctx->Attrs().Get<bool>("usePeepholes")) {
      PADDLE_ENFORCE_EQ(b_dims[1], 7 * frame_size,
                        "The second dimension of Input(Bias) should be "
                        "7 * %d if enable peepholes connection",
                        frame_size);
    } else {
      PADDLE_ENFORCE_EQ(b_dims[1], 4 * frame_size,
                        "The second dimension of Input(Bias) should be "
                        "4 * %d if disable peepholes connection",
                        frame_size);
    }
    framework::DDim out_dims({in_dims[0], frame_size});
    ctx->SetOutputDim("Hidden", out_dims);
    ctx->SetOutputDim("Cell", out_dims);
    ctx->SetOutputDim("BatchGate", in_dims);
    ctx->SetOutputDim("BatchCellPreAct", out_dims);
    ctx->ShareLoD("Input", "Hidden");
    ctx->ShareLoD("Input", "Cell");
  }

 protected:
  framework::OpKernelType GetKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::LoDTensor>("Input")->type()),
        ctx.device_context());
  }
};

class LSTMOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LSTMOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Input",
             "(LoDTensor) the first input is a LodTensor, which support "
             "variable-time length input sequence. The underlying tensor in "
             "this LoDTensor is a matrix with shape (T X 4D), where T is the "
             "total time steps in this mini-batch, D is the hidden size.");
    AddInput("H0",
             "(Tensor, optional) the initial hidden state is an optional "
             "input. This is a tensor with shape (N x D), where N is the "
             "batch size and D is the hidden size.")
        .AsDispensable();
    AddInput("C0",
             "(Tensor, optional) the initial cell state is an optional "
             "input. This is a tensor with shape (N x D), where N is the "
             "batch size. `H0` and `C0` can be NULL but only at the same time")
        .AsDispensable();
    AddInput("Weight",
             "(Tensor) the learnable hidden-hidden weights."
             " - The shape is (D x 4D), where D is the hidden size. "
             " - Weight = {W_ch, W_ih, W_fh, W_oh}");
    AddInput("Bias",
             "(Tensor) the learnable weights, which contains two parts: "
             "input-hidden bias weight and peephole connections weight if "
             "setting `usePeepholes` True. "
             "1. `usePeepholes = False` "
             " - The shape is (1 x 4D). "
             " - Bias = {b_c, b_i, b_f, b_o}."
             "2. `usePeepholes = True` "
             " - The shape is (1 x 7D). "
             " - Bias = {b_c, b_i, b_f, b_o, W_ic, W_fc, W_oc}.")
        .AsDispensable();
    AddOutput("Hidden",
              "(LoDTensor) the hidden state of LSTM operator. "
              "The shape is (T x D), and lod is the same with the `Input`.");
    AddOutput("Cell",
              "(LoDTensor) the cell state of LSTM operator. "
              "The shape is (T x D), and lod is the same with the `Input`.");
    AddOutput("BatchGate",
              "(LoDTensor) This LoDTensor contains input gate, forget gate "
              "and output gate after the nonlinear computation. This "
              "LoDTensor has the same shape as the reorganized input, which "
              "is also be called batch input. The LoD size is 2. The first "
              "LoD is the batch offsets and the second LoD contains the "
              "indexes, which denote the position of reorganized sequence "
              "in the raw input.")
        .AsIntermediate();
    AddOutput("BatchCellPreAct",
              "(LoDTensor) This LoDTensor is obtained in the forward and used "
              "in the backward.")
        .AsIntermediate();
    AddAttr<bool>("usePeepholes",
                  "(bool, default True) "
                  "whether to enable diagonal/peephole connections.")
        .SetDefault(true);
    AddAttr<bool>("isReverse",
                  "(bool, default False) "
                  "whether to compute reversed LSTM.")
        .SetDefault(false);
    AddAttr<std::string>(
        "gateActivation",
        "(string, default sigmoid)"
        "The activation for input gate, forget gate and output "
        "gate, `sigmoid` by default.")
        .SetDefault("sigmoid");
    AddAttr<std::string>("cellActivation",
                         "(string, default tanh)"
                         "The activation for cell output, `tanh` by defalut.")
        .SetDefault("tanh");
    AddAttr<std::string>("candidateActivation",
                         "(string, default tanh)"
                         "The activation for candidate hidden state, "
                         "`tanh` by default.")
        .SetDefault("tanh");
    AddComment(R"DOC(
Long-Short Term Memory (LSTM) Operator.

The defalut implementation is diagonal/peephole connection 
(https://arxiv.org/pdf/1402.1128.pdf), the formula is as follows:

$$
i_t = \sigma(W_{ix}x_{t} + W_{ih}h_{t-1} + W_{ic}c_{t-1} + b_i) \\

f_t = \sigma(W_{fx}x_{t} + W_{fh}h_{t-1} + W_{fc}c_{t-1} + b_f) \\

\tilde{c_t} = act_g(W_{cx}x_t + W_{ch}h_{t-1} + b_c) \\

o_t = \sigma(W_{ox}x_{t} + W_{oh}h_{t-1} + W_{oc}c_t + b_o) \\

c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c_t} \\

h_t = o_t \odot act_h(c_t)
$$

where the W terms denote weight matrices (e.g. \f$W_{xi}\f$ is the matrix
of weights from the input gate to the input), \f$W_{ic}, W_{fc}, W_{oc}\f$
are diagonal weight matrices for peephole connections. In our implementation,
we use vectors to reprenset these diagonal weight matrices. The b terms
denote bias vectors (\f$b_i\f$ is the input gate bias vector), \f$\sigma\f$
is the non-line activations, such as logistic sigmoid function, and
\f$i, f, o\f$ and \f$c\f$ are the input gate, forget gate, output gate,
and cell activation vectors, respectively, all of which have the same size as
the cell output activation vector \f$h\f$.

The \f$\odot\f$ is the element-wise product of the vectors. \f$act_g\f$ and \f$act_h\f$
are the cell input and cell output activation functions and `tanh` is usually
used for them. \f$\tilde{c_t}\f$ is also called candidate hidden state,
which is computed based on the current input and the previous hidden state.

Set usePeepholes False to disable peephole connection 
(http://www.bioinf.jku.at/publications/older/2604.pdf). The formula
is omitted here.

Note that these \f$W_{xi}x_{t}, W_{xf}x_{t}, W_{xc}x_{t}, W_{xo}x_{t}\f$
operations on the input \f$x_{t}\f$ are NOT included in this operator.
Users can choose to use fully-connect operator before LSTM operator.

)DOC");
  }
};

class LSTMGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"),
                   "Input(Input) of LSTM should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Hidden"),
                   "Input(Hidden) of LSTM should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Cell"),
                   "Input(Cell) of LSTM should not be null.");

    PADDLE_ENFORCE(ctx->HasInput("BatchGate"),
                   "Input(BatchGate) of LSTM should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("BatchCellPreAct"),
                   "Input(BatchGate) of LSTM should not be null.");

    auto in_g_name = framework::GradVarName("Input");
    if (ctx->HasOutput(in_g_name))
      ctx->SetOutputDim(in_g_name, ctx->GetInputDim("Input"));

    auto w_g_name = framework::GradVarName("Weight");
    if (ctx->HasOutput(w_g_name))
      ctx->SetOutputDim(w_g_name, ctx->GetInputDim("Weight"));

    auto b_g_name = framework::GradVarName("Bias");
    if (ctx->HasOutput(b_g_name))
      ctx->SetOutputDim(b_g_name, ctx->GetInputDim("Bias"));
  }

 protected:
  framework::OpKernelType GetKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::LoDTensor>("Input")->type()),
        ctx.device_context());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(lstm, ops::LSTMOp, ops::LSTMOpMaker, lstm_grad, ops::LSTMGradOp);
REGISTER_OP_CPU_KERNEL(lstm, ops::LSTMKernel<paddle::platform::CPUPlace, float>,
                       ops::LSTMKernel<paddle::platform::CPUPlace, double>);
REGISTER_OP_CPU_KERNEL(lstm_grad,
                       ops::LSTMGradKernel<paddle::platform::CPUPlace, float>,
                       ops::LSTMGradKernel<paddle::platform::CPUPlace, double>);
