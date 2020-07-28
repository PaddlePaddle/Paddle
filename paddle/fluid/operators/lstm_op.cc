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

#include "paddle/fluid/operators/lstm_op.h"
#include <memory>
#include <string>

namespace paddle {
namespace operators {

class LSTMOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "LSTM");
    OP_INOUT_CHECK(ctx->HasInput("Weight"), "Input", "Weight", "LSTM");
    OP_INOUT_CHECK(ctx->HasInput("Bias"), "Input", "Bias", "LSTM");

    OP_INOUT_CHECK(ctx->HasOutput("Hidden"), "Output", "Hidden", "LSTM");
    OP_INOUT_CHECK(ctx->HasOutput("Cell"), "Output", "Cell", "LSTM");
    OP_INOUT_CHECK(ctx->HasOutput("BatchGate"), "Output", "BatchGate", "LSTM");
    OP_INOUT_CHECK(ctx->HasOutput("BatchCellPreAct"), "Output",
                   "BatchCellPreAct", "LSTM");

    auto in_dims = ctx->GetInputDim("Input");
    PADDLE_ENFORCE_EQ(
        in_dims.size(), 2,
        platform::errors::InvalidArgument(
            "Input(X)'s rank must be 2, but received %d.", in_dims.size()));

    if (ctx->HasInput("H0")) {
      PADDLE_ENFORCE_EQ(
          ctx->HasInput("C0"), true,
          platform::errors::NotFound("Input(Cell) and Input(Hidden) of LSTM "
                                     "should not be null at the same time."));
      auto h_dims = ctx->GetInputDim("H0");
      auto c_dims = ctx->GetInputDim("C0");
      PADDLE_ENFORCE_EQ(h_dims, c_dims,
                        platform::errors::InvalidArgument(
                            "The dimension of Input(H0) and Input(C0) should "
                            "be the same, but received [%s] (H0) vs [%s] (C0).",
                            h_dims, c_dims));
    }

    int frame_size = in_dims[1] / 4;
    auto w_dims = ctx->GetInputDim("Weight");
    PADDLE_ENFORCE_EQ(
        w_dims.size(), 2,
        platform::errors::InvalidArgument(
            "The rank of Input(Weight) should be 2, but received %d.",
            w_dims.size()));
    PADDLE_ENFORCE_EQ(w_dims[0], frame_size,
                      platform::errors::InvalidArgument(
                          "The first dimension of Input(Weight) should be %d, "
                          "but received %d.",
                          frame_size, w_dims[0]));
    PADDLE_ENFORCE_EQ(w_dims[1], 4 * frame_size,
                      platform::errors::InvalidArgument(
                          "The second dimension of Input(Weight) should be 4 * "
                          "%d, but received %d.",
                          frame_size, w_dims[1]));

    auto b_dims = ctx->GetInputDim("Bias");
    PADDLE_ENFORCE_EQ(
        b_dims.size(), 2,
        platform::errors::InvalidArgument(
            "The rank of Input(Bias) should be 2, but received %d.",
            b_dims.size()));
    PADDLE_ENFORCE_EQ(
        b_dims[0], 1,
        platform::errors::InvalidArgument(
            "The first dimension of Input(Bias) should be 1, but received %d.",
            b_dims[0]));

    if (ctx->Attrs().Get<bool>("use_peepholes")) {
      PADDLE_ENFORCE_EQ(
          b_dims[1], 7 * frame_size,
          platform::errors::InvalidArgument(
              "The second dimension of Input(Bias) should be 7 * %d if enable "
              "peepholes connection, but received %d.",
              frame_size, b_dims[1]));
    } else {
      PADDLE_ENFORCE_EQ(
          b_dims[1], 4 * frame_size,
          platform::errors::InvalidArgument(
              "The second dimension of Input(Bias) should be 4 * %d if disable "
              "peepholes connection, but received %d.",
              frame_size, b_dims[1]));
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
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
        ctx.device_context());
  }
};

class LSTMOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
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
             "batch size. `H0` and `C0` can be NULL but only at the same time.")
        .AsDispensable();
    AddInput("Weight",
             "(Tensor) the learnable hidden-hidden weights."
             " - The shape is (D x 4D), where D is the hidden size. "
             " - Weight = {W_ch, W_ih, W_fh, W_oh}");
    AddInput("Bias",
             "(Tensor) the learnable weights, which contains two parts: "
             "input-hidden bias weight and peephole connections weight if "
             "setting `use_peepholes` True. "
             "1. `use_peepholes = False` "
             " - The shape is (1 x 4D). "
             " - Bias = {b_c, b_i, b_f, b_o}."
             "2. `use_peepholes = True` "
             " - The shape is (1 x 7D). "
             " - Bias = {b_c, b_i, b_f, b_o, W_ic, W_fc, W_oc}.");
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
    AddAttr<bool>("use_peepholes",
                  "(bool, default: True) "
                  "whether to enable diagonal/peephole connections.")
        .SetDefault(true);
    AddAttr<bool>("is_reverse",
                  "(bool, default: False) "
                  "whether to compute reversed LSTM.")
        .SetDefault(false);
    AddAttr<std::string>(
        "gate_activation",
        "(string, default: sigmoid)"
        "The activation for input gate, forget gate and output "
        "gate, `sigmoid` by default.")
        .SetDefault("sigmoid")
        .InEnum({"sigmoid", "tanh", "relu", "identity"});
    AddAttr<std::string>("cell_activation",
                         "(string, default: tanh)"
                         "The activation for cell output, `tanh` by default.")
        .SetDefault("tanh")
        .InEnum({"sigmoid", "tanh", "relu", "identity"});
    AddAttr<std::string>("candidate_activation",
                         "(string, default: tanh)"
                         "The activation for candidate hidden state, "
                         "`tanh` by default.")
        .SetDefault("tanh")
        .InEnum({"sigmoid", "tanh", "relu", "identity"});
    AddComment(R"DOC(
Long-Short Term Memory (LSTM) Operator.

The default implementation is diagonal/peephole connection
(https://arxiv.org/pdf/1402.1128.pdf), the formula is as follows:

$$ i_t = \\sigma(W_{ix}x_{t} + W_{ih}h_{t-1} + W_{ic}c_{t-1} + b_i) $$

$$ f_t = \\sigma(W_{fx}x_{t} + W_{fh}h_{t-1} + W_{fc}c_{t-1} + b_f) $$

$$ \\tilde{c_t} = act_g(W_{cx}x_t + W_{ch}h_{t-1} + b_c) $$

$$ o_t = \\sigma(W_{ox}x_{t} + W_{oh}h_{t-1} + W_{oc}c_t + b_o) $$

$$ c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c_t} $$

$$ h_t = o_t \\odot act_h(c_t) $$

- W terms denote weight matrices (e.g. $W_{xi}$ is the matrix
  of weights from the input gate to the input), $W_{ic}, W_{fc}, W_{oc}$
  are diagonal weight matrices for peephole connections. In our implementation,
  we use vectors to represent these diagonal weight matrices.
- The b terms denote bias vectors ($b_i$ is the input gate bias vector).
- $\sigma$ is the non-line activations, such as logistic sigmoid function.
- $i, f, o$ and $c$ are the input gate, forget gate, output gate,
  and cell activation vectors, respectively, all of which have the same size as
  the cell output activation vector $h$.
- The $\odot$ is the element-wise product of the vectors.
- $act_g$ and $act_h$ are the cell input and cell output activation functions
  and `tanh` is usually used for them.
- $\tilde{c_t}$ is also called candidate hidden state,
  which is computed based on the current input and the previous hidden state.

Set `use_peepholes` False to disable peephole connection. The formula
is omitted here, please refer to the paper
http://www.bioinf.jku.at/publications/older/2604.pdf for details.

Note that these $W_{xi}x_{t}, W_{xf}x_{t}, W_{xc}x_{t}, W_{xo}x_{t}$
operations on the input $x_{t}$ are NOT included in this operator.
Users can choose to use fully-connect operator before LSTM operator.

)DOC");
  }
};

class LSTMGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "LSTM@Grad");
    OP_INOUT_CHECK(ctx->HasInput("Hidden"), "Input", "Hidden", "LSTM@Grad");
    OP_INOUT_CHECK(ctx->HasInput("Cell"), "Input", "Cell", "LSTM@Grad");
    OP_INOUT_CHECK(ctx->HasInput("Weight"), "Input", "Weight", "LSTM@Grad");
    OP_INOUT_CHECK(ctx->HasInput("Bias"), "Input", "Bias", "LSTM@Grad");

    OP_INOUT_CHECK(ctx->HasInput("BatchGate"), "Input", "BatchGate",
                   "LSTM@Grad");
    OP_INOUT_CHECK(ctx->HasInput("BatchCellPreAct"), "Input", "BatchCellPreAct",
                   "LSTM@Grad");

    auto SetOutGradDim = [&ctx](const std::string& name) {
      auto g_name = framework::GradVarName(name);
      if (ctx->HasOutput(g_name))
        ctx->SetOutputDim(g_name, ctx->GetInputDim(name));
    };

    SetOutGradDim("Input");
    SetOutGradDim("Weight");
    SetOutGradDim("Bias");
    SetOutGradDim("H0");
    SetOutGradDim("C0");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
        ctx.device_context());
  }
};

template <typename T>
class LSTMGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("lstm_grad");
    op->SetAttrMap(this->Attrs());
    op->SetInput("Input", this->Input("Input"));
    op->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));

    if (this->HasInput("H0")) {
      op->SetInput("H0", this->Input("H0"));
      op->SetOutput(framework::GradVarName("H0"), this->InputGrad("H0"));
    }

    if (this->HasInput("C0")) {
      op->SetInput("C0", this->Input("C0"));
      op->SetOutput(framework::GradVarName("C0"), this->InputGrad("C0"));
    }

    op->SetInput("Weight", this->Input("Weight"));
    op->SetOutput(framework::GradVarName("Weight"), this->InputGrad("Weight"));

    op->SetInput("Bias", this->Input("Bias"));
    op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));

    op->SetInput("Cell", this->Output("Cell"));

    op->SetInput("Hidden", this->Output("Hidden"));
    op->SetInput(framework::GradVarName("Hidden"), this->OutputGrad("Hidden"));

    op->SetInput("BatchGate", this->Output("BatchGate"));
    op->SetInput("BatchCellPreAct", this->Output("BatchCellPreAct"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(lstm, ops::LSTMOp, ops::LSTMOpMaker,
                  ops::LSTMGradOpMaker<paddle::framework::OpDesc>,
                  ops::LSTMGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(lstm_grad, ops::LSTMGradOp);
REGISTER_OP_CPU_KERNEL(
    lstm, ops::LSTMKernel<paddle::platform::CPUDeviceContext, float>,
    ops::LSTMKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    lstm_grad, ops::LSTMGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::LSTMGradKernel<paddle::platform::CPUDeviceContext, double>);
