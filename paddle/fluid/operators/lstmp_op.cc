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

#include "paddle/fluid/operators/lstmp_op.h"

#include <memory>
#include <string>

namespace paddle {
namespace operators {

class LSTMPOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "LSTMP");
    OP_INOUT_CHECK(ctx->HasInput("Weight"), "Input", "Weight", "LSTMP");
    OP_INOUT_CHECK(ctx->HasInput("ProjWeight"), "Input", "ProjWeight", "LSTMP");
    OP_INOUT_CHECK(ctx->HasInput("Bias"), "Input", "Bias", "LSTMP");

    OP_INOUT_CHECK(
        ctx->HasOutput("Projection"), "Output", "Projection", "LSTMP");
    OP_INOUT_CHECK(ctx->HasOutput("Cell"), "Output", "Cell", "LSTMP");
    OP_INOUT_CHECK(ctx->HasOutput("BatchGate"), "Output", "BatchGate", "LSTMP");
    OP_INOUT_CHECK(ctx->HasOutput("BatchCellPreAct"),
                   "Output",
                   "BatchCellPreAct",
                   "LSTMP");
    OP_INOUT_CHECK(
        ctx->HasOutput("BatchHidden"), "Output", "BatchHidden", "LSTMP");

    auto in_dims = ctx->GetInputDim("Input");

    PADDLE_ENFORCE_EQ(
        in_dims.size(),
        2,
        platform::errors::InvalidArgument(
            "Input(X)'s rank of LSTMP operator must be 2, but received %d.",
            in_dims.size()));

    int frame_size = in_dims[1] / 4;
    auto w_dims = ctx->GetInputDim("Weight");
    auto proj_dims = ctx->GetInputDim("ProjWeight");
    PADDLE_ENFORCE_EQ(
        w_dims.size(),
        2,
        platform::errors::InvalidArgument(
            "The rank of Input(Weight) should be 2, but received %d.",
            w_dims.size()));
    PADDLE_ENFORCE_EQ(
        w_dims[0],
        proj_dims[1],
        platform::errors::InvalidArgument(
            "The first dimension of Input(Weight) and the second dimension of "
            "Input(ProjWeight) should be the same, but received %d vs %d.",
            w_dims[0],
            proj_dims[1]));
    PADDLE_ENFORCE_EQ(w_dims[1],
                      4 * frame_size,
                      platform::errors::InvalidArgument(
                          "The second dimension of Input(Weight) should be 4 * "
                          "%d, but received %d.",
                          frame_size,
                          w_dims[1]));

    PADDLE_ENFORCE_EQ(
        proj_dims.size(),
        2,
        platform::errors::InvalidArgument(
            "The rank of Input(ProjWeight) should be 2, but received %d.",
            proj_dims.size()));
    PADDLE_ENFORCE_EQ(proj_dims[0],
                      frame_size,
                      platform::errors::InvalidArgument(
                          "The first dimension of Input(ProjWeight) should be "
                          "%d, but received %d.",
                          frame_size,
                          proj_dims[0]));

    if (ctx->HasInput("H0")) {
      PADDLE_ENFORCE_EQ(
          ctx->HasInput("C0"),
          true,
          platform::errors::NotFound("Input(C0) of LSTMP operator should not "
                                     "be null after Input(H0) provided."));
    }

    auto b_dims = ctx->GetInputDim("Bias");
    PADDLE_ENFORCE_EQ(
        b_dims.size(),
        2,
        platform::errors::InvalidArgument(
            "The rank of Input(Bias) should be 2, but received %d.",
            b_dims.size()));
    PADDLE_ENFORCE_EQ(
        b_dims[0],
        1,
        platform::errors::InvalidArgument(
            "The first dimension of Input(Bias) should be 1, but received %d.",
            b_dims[0]));

    if (ctx->Attrs().Get<bool>("use_peepholes")) {
      PADDLE_ENFORCE_EQ(
          b_dims[1],
          7 * frame_size,
          platform::errors::InvalidArgument(
              "The second dimension of Input(Bias) should be 7 * %d if enable "
              "peepholes connection, but received %d.",
              frame_size,
              b_dims[1]));
    } else {
      PADDLE_ENFORCE_EQ(
          b_dims[1],
          4 * frame_size,
          platform::errors::InvalidArgument(
              "The second dimension of Input(Bias) should be 4 * %d if disable "
              "peepholes connection, but received %d.",
              frame_size,
              b_dims[1]));
    }

    framework::DDim out_dims({in_dims[0], frame_size});
    framework::DDim proj_out_dims({in_dims[0], proj_dims[1]});
    ctx->SetOutputDim("Projection", proj_out_dims);
    ctx->SetOutputDim("Cell", out_dims);
    ctx->SetOutputDim("BatchGate", in_dims);
    ctx->SetOutputDim("BatchCellPreAct", out_dims);
    ctx->SetOutputDim("BatchHidden", out_dims);
    ctx->ShareLoD("Input", "Projection");
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

class LSTMPOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Input",
             "(LoDTensor) the input for sequence data, which supports "
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
             "batch size. `C0` should not be null if `H0` provided.")
        .AsDispensable();
    AddInput("Weight",
             "(Tensor) the learnable hidden-hidden weights."
             " - The shape is (P x 4D), where P is the projection layer size "
             "and  D is the hidden size."
             " - Weight = {W_cr, W_ir, W_fr, W_or}");
    AddInput("ProjWeight",
             "(Tensor) the learnable weight of the projection layer."
             " - The shape is (D x P), where P is the recurrent projection "
             "layer size and  D is the hidden size."
             " - ProjWeight = {W_rh}");
    AddInput("Bias",
             "(Tensor) the learnable biases, which contains two parts: "
             "input-hidden biases and peephole connections weights if "
             "setting `use_peepholes` to `True`. "
             "1. `use_peepholes = False` "
             " - The shape is (1 x 4D). "
             " - Bias = {b_c, b_i, b_f, b_o}."
             "2. `use_peepholes = True` "
             " - The shape is (1 x 7D). "
             " - Bias = {b_c, b_i, b_f, b_o, W_ic, W_fc, W_oc}.");
    AddOutput("Projection",
              "(LoDTensor) the projection of the hidden state of LSTMP "
              "operator. The shape is (T x P), and LoD is the same with the "
              "`Input`.");
    AddOutput("Cell",
              "(LoDTensor) the cell state of LSTMP operator. "
              "The shape is (T x D), and lod is the same with the `Input`.");
    AddOutput("BatchGate",
              "(LoDTensor) This LoDTensor contains input gate, forget gate "
              "and output gate after the activations. This LoDTensor has the "
              "same shape as the reorganized input, which is also be called "
              "batch input. The LoD size is 2. The first-level LoD is the "
              "batch offsets and the second contains the indices, which "
              "denotes the position of reorganized sequence in the raw input.")
        .AsIntermediate();
    AddOutput("BatchCellPreAct",
              "(LoDTensor) the pre-activation cell state reorganized in batch. "
              "This LoDTensor is obtained in the forward and used in the "
              "backward.")
        .AsIntermediate();
    AddOutput("BatchHidden",
              "(LoDTensor) the hidden state reorganized in batch. "
              "This LoDTensor is obtained in the forward and used in the "
              "backward.")
        .AsIntermediate();
    AddAttr<bool>("use_peepholes",
                  "(bool, default: True) "
                  "whether to enable diagonal/peephole connections.")
        .SetDefault(true);
    AddAttr<bool>("is_reverse",
                  "(bool, default: False) "
                  "whether to compute reversed LSTMP.")
        .SetDefault(false);
    AddAttr<float>("cell_clip",
                   "(float, default: 0.0) "
                   "Clip for Tensor for cell state tensor when clip value is "
                   "greater than 0.0")
        .SetDefault(0.0);
    AddAttr<float>("proj_clip",
                   "(float, default: 0.0) "
                   "Clip for Tensor for projection tensor when clip value is "
                   "greater than 0.0")
        .SetDefault(0.0);
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
    AddAttr<std::string>("proj_activation",
                         "(string, default: tanh)"
                         "The activation for projection output, "
                         "`tanh` by default.")
        .SetDefault("tanh")
        .InEnum({"sigmoid", "tanh", "relu", "identity"});
    AddComment(R"DOC(
Long-Short Term Memory with recurrent Projection layer (LSTMP) Operator.

LSTMP has a separate projection layer after the LSTM layer, projecting the
original hidden state to a lower-dimensional one, which is proposed to reduce
the number of total parameters and furthermore computational complexity for
the LSTM, espeacially for the case that the size of output units is relative
large (https://research.google.com/pubs/archive/43905.pdf).

The formula is as follows:

$$
i_t = \sigma(W_{ix}x_{t} + W_{ir}r_{t-1} + W_{ic}c_{t-1} + b_i) \\

f_t = \sigma(W_{fx}x_{t} + W_{fr}r_{t-1} + W_{fc}c_{t-1} + b_f) \\

\tilde{c_t} = act_g(W_{cx}x_t + W_{cr}r_{t-1} + b_c) \\

o_t = \sigma(W_{ox}x_{t} + W_{or}r_{t-1} + W_{oc}c_t + b_o) \\

c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c_t} \\

h_t = o_t \odot act_h(c_t) \\

r_t = \overline{act_h}(W_{rh}h_t)
$$

where the W terms denote weight matrices (e.g. $W_{xi}$ is the matrix
of weights from the input gate to the input), $W_{ic}, W_{fc}, W_{oc}$
are diagonal weight matrices for peephole connections. In our implementation,
we use vectors to represent these diagonal weight matrices. The b terms
denote bias vectors ($b_i$ is the input gate bias vector), $\sigma$
is the activation, such as logistic sigmoid function, and
$i, f, o$ and $c$ are the input gate, forget gate, output gate,
and cell activation vectors, respectively, all of which have the same size as
the cell output activation vector $h$. Here $h$ is usually called the hidden
state and $r$ denotes its recurrent projection. And $\tilde{c_t}$ is also
called the candidate hidden state, whose computation is based on the current
input and previous hidden state.

The $\odot$ is the element-wise product of the vectors. $act_g$ and $act_h$
are the cell input and cell output activation functions and `tanh` is usually
used for them. $\overline{act_h}$ is the activation function for the
projection output, usually using `identity` or same as $act_h$.

Note that these $W_{xi}x_{t}, W_{xf}x_{t}, W_{xc}x_{t}, W_{xo}x_{t}$
operations on the input $x_{t}$ are NOT included in this operator.
Users can choose to use fully-connected operator before LSTMP operator.

)DOC");
  }
};

template <typename T>
class LSTMPGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("lstmp_grad");
    grad_op->SetInput("Weight", this->Input("Weight"));
    grad_op->SetInput("ProjWeight", this->Input("ProjWeight"));
    grad_op->SetInput("Bias", this->Input("Bias"));

    grad_op->SetInput("Projection", this->Output("Projection"));
    grad_op->SetInput("Cell", this->Output("Cell"));
    grad_op->SetInput("BatchGate", this->Output("BatchGate"));
    grad_op->SetInput("BatchCellPreAct", this->Output("BatchCellPreAct"));
    grad_op->SetInput("BatchHidden", this->Output("BatchHidden"));
    grad_op->SetInput("H0", this->Input("H0"));
    grad_op->SetInput("C0", this->Input("C0"));

    grad_op->SetInput(framework::GradVarName("Projection"),
                      this->OutputGrad("Projection"));

    grad_op->SetOutput(framework::GradVarName("Input"),
                       this->InputGrad("Input"));
    grad_op->SetOutput(framework::GradVarName("Weight"),
                       this->InputGrad("Weight"));
    grad_op->SetOutput(framework::GradVarName("ProjWeight"),
                       this->InputGrad("ProjWeight"));
    grad_op->SetOutput(framework::GradVarName("Bias"), this->InputGrad("Bias"));
    grad_op->SetOutput(framework::GradVarName("H0"), this->InputGrad("H0"));
    grad_op->SetOutput(framework::GradVarName("C0"), this->InputGrad("C0"));

    grad_op->SetAttrMap(this->Attrs());
  }
};

class LSTMPGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("Projection"), "Input", "Projection", "LSTMP@Grad");
    OP_INOUT_CHECK(ctx->HasInput("Cell"), "Input", "Cell", "LSTMP@Grad");
    OP_INOUT_CHECK(ctx->HasInput("Weight"), "Input", "Weight", "LSTMP@Grad");
    OP_INOUT_CHECK(
        ctx->HasInput("ProjWeight"), "Input", "ProjWeight", "LSTMP@Grad");
    OP_INOUT_CHECK(ctx->HasInput("Bias"), "Input", "Bias", "LSTMP@Grad");

    OP_INOUT_CHECK(
        ctx->HasInput("BatchGate"), "Input", "BatchGate", "LSTMP@Grad");
    OP_INOUT_CHECK(ctx->HasInput("BatchCellPreAct"),
                   "Input",
                   "BatchCellPreAct",
                   "LSTMP@Grad");

    auto SetOutGradDim = [&ctx](const std::string& name) {
      auto g_name = framework::GradVarName(name);
      if (ctx->HasOutput(g_name))
        ctx->SetOutputDim(g_name, ctx->GetInputDim(name));
    };

    ctx->SetOutputDim(framework::GradVarName("Input"),
                      ctx->GetInputDim("BatchGate"));
    SetOutGradDim("Weight");
    SetOutGradDim("ProjWeight");
    SetOutGradDim("Bias");
    SetOutGradDim("H0");
    SetOutGradDim("C0");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "BatchGate"),
        ctx.device_context());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(lstmp,
                  ops::LSTMPOp,
                  ops::LSTMPOpMaker,
                  ops::LSTMPGradMaker<paddle::framework::OpDesc>,
                  ops::LSTMPGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(lstmp_grad, ops::LSTMPGradOp);
REGISTER_OP_CPU_KERNEL(lstmp,
                       ops::LSTMPKernel<phi::CPUContext, float>,
                       ops::LSTMPKernel<phi::CPUContext, double>);
REGISTER_OP_CPU_KERNEL(lstmp_grad,
                       ops::LSTMPGradKernel<phi::CPUContext, float>,
                       ops::LSTMPGradKernel<phi::CPUContext, double>);
