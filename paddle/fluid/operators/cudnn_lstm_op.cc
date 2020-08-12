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

#include <memory>
#include <string>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class CudnnLSTMOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "CudnnLSTM");
    OP_INOUT_CHECK(ctx->HasInput("W"), "Input", "W", "CudnnLSTM");
    OP_INOUT_CHECK(ctx->HasInput("InitH"), "Input", "InitH", "CudnnLSTM");
    OP_INOUT_CHECK(ctx->HasInput("InitC"), "Input", "InitC", "CudnnLSTM");

    OP_INOUT_CHECK(ctx->HasOutput("Reserve"), "Output", "Reserve", "CudnnLSTM");
    OP_INOUT_CHECK(ctx->HasOutput("StateOut"), "Output", "StateOut",
                   "CudnnLSTM");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "CudnnLSTM");
    OP_INOUT_CHECK(ctx->HasOutput("LastH"), "Output", "LastH", "CudnnLSTM");
    OP_INOUT_CHECK(ctx->HasOutput("LastC"), "Output", "LastC", "CudnnLSTM");

    auto in_dims = ctx->GetInputDim("Input");
    auto init_dims = ctx->GetInputDim("InitH");
    PADDLE_ENFORCE_EQ(in_dims.size(), 3,
                      platform::errors::InvalidArgument(
                          "The rank of Input in CudnnLSTM  must be 3. But "
                          "received Input's rank is %d.",
                          in_dims.size()));
    PADDLE_ENFORCE_EQ(init_dims.size(), 3,
                      platform::errors::InvalidArgument(
                          "The rank of InitH in CudnnLSTM  must be 3. But "
                          "received InitH's rank is %d.",
                          init_dims.size()));

    PADDLE_ENFORCE_EQ(in_dims[1], init_dims[1],
                      platform::errors::InvalidArgument(
                          "The in_dims[1] (Input dims) and init_dims[1] (InitH "
                          "dims) should be equal. But "
                          "received in_dims[1] is %d and init_dims[1] is %d.",
                          in_dims[1], init_dims[1]));
    PADDLE_ENFORCE_EQ(in_dims[2], init_dims[2],
                      platform::errors::InvalidArgument(
                          "The in_dims[2] (Input dims) and init_dims[2] (InitH "
                          "dims) should be equal. But "
                          "received in_dims[2] is %d and init_dims[2] is %d.",
                          in_dims[2], init_dims[2]));

    auto out_dims = in_dims;
    auto hidden_size = ctx->Attrs().Get<int>("hidden_size");
    bool is_bidirec = ctx->Attrs().Get<bool>("is_bidirec");
    out_dims[2] = is_bidirec ? hidden_size * 2 : hidden_size;

    auto last_dims = init_dims;
    last_dims[0] = is_bidirec ? last_dims[0] * 2 : last_dims[0];
    ctx->SetOutputDim("Out", out_dims);
    ctx->SetOutputDim("LastH", last_dims);
    ctx->SetOutputDim("LastC", last_dims);
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "Input"),
        ctx.device_context());
  }
};

class CudnnLSTMOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput(
        "Input",
        "(Tensor) RNN input tensor, which support variable-time length input "
        "sequence."
        "The shape of the Tensor MUST be ( seq_len * batch_size * input_size)"
        "seq_len is the total time step in this mini-batch (CAN be change in "
        "different batch)"
        "batch_size is the instance number of this batch"
        "input_size is the hidden size of the input."
        "input_hidden_size and the hidden_size in the next may not be same");
    AddInput("InitH",
             "(Tensor) the initial hidden state of the LSTM"
             "input. This is a tensor with shape (num_layers x batch_size x "
             "hidden_size)"
             "and When is_bidirec is True, the shape will be (num_layers*2 x "
             "batch_size x hidden_size)");
    AddInput("InitC",
             "(Tensor) the initial cell state of the LSTm "
             "input. This is a tensor with shape (num_layers x batch_size x "
             "hidden_size)"
             "and When is_bidirec is True, the shape will be (num_layers*2 x "
             "batch_size x hidden_size)");
    AddInput("W",
             "(Tensor) the learnable hidden-hidden weights."
             " The shape is (N), where N is total weight size of the LSTM. "
             " cudnn concatenate all the weight to one Tensor");
    AddOutput("Reserve",
              "(Tensor, a temporary output Tensor to store the reserve_data "
              "of cudnn kernel.")
        .AsIntermediate();
    AddOutput("StateOut",
              "Share memory with State. "
              "Store the global drop state when training");
    AddOutput("Out",
              "(Tensor) the hidden state of LSTM operator. "
              "The shape is ( seq_len x batch_size x hidden_size) if "
              "is_bidirec is False"
              "and When is_bidirec is True, the shape will be ( seq_len x "
              "batch_size x hidden_size * 2) ");
    AddOutput("LastH",
              "(Tensor) the hidden state of the last step. "
              "The shape is ( num_layers x batch_size x hidden_size) if "
              "is_bidirec is False"
              "and When is_bidirec is True, the shape will be (num_layers*2 x "
              "batch_size x hidden_size)");
    AddOutput("LastC",
              "(Tensor) the cell state of the last step"
              "The shape is ( num_layers x batch_size x hidden_size) if "
              "is_bidirec is False"
              "and When is_bidirect is True, the shape will be (num_layers*2 x "
              "batch_size x hidden_size*2)");
    AddAttr<float>(
        "dropout_prob",
        "dropout prob of the dropout op"
        "the dropout ONLY work between lstm layers, not between time steps"
        "There is no dropout work on the Out tensor")
        .SetDefault(0.0);
    AddAttr<bool>("is_bidirec",
                  "is_bidirec"
                  "if it is bidirectional rnn"
                  "The will affect the shape of the Out, LastH, and LastC")
        .SetDefault(false);
    AddAttr<int>("input_size", "input size ot the Input Tensor").SetDefault(10);
    AddAttr<int>("hidden_size", "hidden size of the LSTM").SetDefault(100);
    AddAttr<int>("num_layers", "the total layer number of the LSTM")
        .SetDefault(1);
    AddAttr<bool>("is_test", "True if in test phase.").SetDefault(false);
    AddAttr<int>("seed", "seed to used if fix_seed is True").SetDefault(0);
    AddComment(R"DOC(
CUDNN LSTM implementation

A four-gate Long Short-Term Memory network with no peephole connections.
In the forward pass the output ht and cell output ct for a given iteration can be computed from the recurrent input ht-1, 
the cell input ct-1 and the previous layer input xt given matrices W, R and biases bW, bR from the following equations:

$$ i_t = sigmoid(W_{ix}x_{t} + W_{ih}h_{t-1} + bx_i + bh_i) $$

$$ f_t = sigmoid(W_{fx}x_{t} + W_{fh}h_{t-1} + bx_f + bh_f) $$

$$ o_t = sigmoid(W_{ox}x_{t} + W_{oh}h_{t-1} + bx_o + bh_o) $$

$$ \\tilde{c_t} = tanh(W_{cx}x_t + W_{ch}h_{t-1} + bx_c + bh_c) $$

$$ c_t = f_t \\odot c_{t-1} + i_t \\odot \\tilde{c_t} $$

$$ h_t = o_t \\odot tanh(c_t) $$

- W terms denote weight matrices (e.g. $W_{ix}$ is the matrix
  of weights from the input gate to the input)
- The b terms denote bias vectors ($bx_i$ and $bh_i$ are the input gate bias vector).
- sigmoid is the logistic sigmoid function.
- $i, f, o$ and $c$ are the input gate, forget gate, output gate,
  and cell activation vectors, respectively, all of which have the same size as
  the cell output activation vector $h$.
- The $\odot$ is the element-wise product of the vectors.
- `tanh` is the activation functions.
- $\tilde{c_t}$ is also called candidate hidden state,
  which is computed based on the current input and the previous hidden state.

Where sigmoid is the sigmoid operator: sigmoid(x) = 1 / (1 + e^-x), * represents a point-wise multiplication, 
X represensts a matrix multiplication


)DOC");
  }
};

class CudnnLSTMGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Input"), "Input", "Input", "CudnnLSTMGrad");
    OP_INOUT_CHECK(ctx->HasInput("W"), "Input", "W", "CudnnLSTMGrad");
    OP_INOUT_CHECK(ctx->HasInput("InitH"), "Input", "InitH", "CudnnLSTMGrad");
    OP_INOUT_CHECK(ctx->HasInput("InitC"), "Input", "InitC", "CudnnLSTMGrad");

    auto SetOutGradDim = [&ctx](const std::string& name) {
      auto g_name = framework::GradVarName(name);
      if (ctx->HasOutput(g_name)) {
        ctx->SetOutputDim(g_name, ctx->GetInputDim(name));
      }
    };

    SetOutGradDim("Input");
    SetOutGradDim("W");
    SetOutGradDim("InitH");
    SetOutGradDim("InitC");
  }
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(OperatorWithKernel::IndicateVarDataType(
                                       ctx, framework::GradVarName("Out")),
                                   ctx.device_context());
  }
};

template <typename T>
class CudnnLSTMGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("cudnn_lstm_grad");
    op->SetInput("Input", this->Input("Input"));
    op->SetInput("InitH", this->Input("InitH"));
    op->SetInput("InitC", this->Input("InitC"));
    op->SetInput("W", this->Input("W"));
    op->SetInput("Reserve", this->Output("Reserve"));
    op->SetInput("StateOut", this->Output("StateOut"));
    op->SetInput("Out", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetInput(framework::GradVarName("LastC"), this->OutputGrad("LastC"));
    op->SetInput(framework::GradVarName("LastH"), this->OutputGrad("LastH"));

    op->SetOutput(framework::GradVarName("Input"), this->InputGrad("Input"));
    op->SetOutput(framework::GradVarName("W"), this->InputGrad("W"));
    op->SetOutput(framework::GradVarName("InitH"), this->InputGrad("InitH"));
    op->SetOutput(framework::GradVarName("InitC"), this->InputGrad("InitC"));
    op->SetAttrMap(this->Attrs());
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
REGISTER_OPERATOR(cudnn_lstm, ops::CudnnLSTMOp, ops::CudnnLSTMOpMaker,
                  ops::CudnnLSTMGradOpMaker<paddle::framework::OpDesc>,
                  ops::CudnnLSTMGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(cudnn_lstm_grad, ops::CudnnLSTMGradOp);

REGISTER_OP_CPU_KERNEL(cudnn_lstm, ops::NotImpleKernel<float>);
REGISTER_OP_CPU_KERNEL(cudnn_lstm_grad, ops::NotImpleKernel<float>);
