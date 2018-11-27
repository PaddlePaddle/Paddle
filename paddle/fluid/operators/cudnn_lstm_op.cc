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

#include "paddle/fluid/operators/cudnn_lstm_op.h"
#include <string>

#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/cudnn_helper.h"
#endif

namespace paddle {
namespace operators {

class CudnnLSTMOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"),
                   "Input(Input) of LSTM should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("W"),
                   "Input(Weight) of LSTM should not be null.");

    PADDLE_ENFORCE(ctx->HasInput("InitH"),
                   "Input(init_h) of LSTM should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("InitC"),
                   "Input(init_c) of LSTM should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Cache"),
                   "Input(Cache) of LSTM should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of LSTM should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("last_h"),
                   "Output(last_h) of LSTM should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("last_c"),
                   "Output(last_c) of LSTM should not be null.");

    auto in_dims = ctx->GetInputDim("Input");
    PADDLE_ENFORCE_EQ(in_dims.size(), 3, "Input(X)'s rank must be 3.");

    ctx->SetOutputDim("Out", ctx->GetInputDim("Input"));
    ctx->SetOutputDim("last_h", ctx->GetInputDim("InitH"));
    ctx->SetOutputDim("last_c", ctx->GetInputDim("InitC"));
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
    AddInput("Cache",
             "The cache of dropout op, a RAW type variable including random "
             "number generator states and some descriptors, which is used in "
             "cudnn kernel.")
        .AsDispensable();
    AddOutput("Out",
              "(Tensor) the hidden state of LSTM operator. "
              "The shape is ( seq_len x batch_size x hidden_size) if "
              "is_bidirec is False"
              "and When is_bidirec is True, the shape will be ( seq_len x "
              "batch_size x hidden_size * 2) ");
    AddOutput("last_h",
              "(Tensor) the hidden state of the last step. "
              "The shape is ( num_layers x batch_size x hidden_size) if "
              "is_bidirec is False"
              "and When is_bidirec is True, the shape will be (num_layers*2 x "
              "batch_size x hidden_size)");
    AddOutput("last_c",
              "(Tensor) the cell state of the last step"
              "The shape is ( num_layers x batch_size x hidden_size) if "
              "is_bidirec is False"
              "and When is_bidirect is True, the shape will be (num_layers*2 x "
              "batch_size x hidden_size*2)");
    AddAttr<int>("max_len",
                 "max length of the LSTM op"
                 "the first dim of the Input can NOT be greater than max_len")
        .SetDefault(20);
    AddAttr<float>(
        "dropout_prob",
        "dropout prob of the dropout op"
        "the dropout ONLY work between lstm layers, not between time steps"
        "There is no dropout work on the Out tensor")
        .SetDefault(0.0);
    AddAttr<bool>("is_bidirec",
                  "is_bidirec"
                  "if it is bidirection rnn"
                  "The will affect the shape of the Out, last_h, and last_c")
        .SetDefault(false);
    AddAttr<int>("input_size", "input size ot the Input Tensor").SetDefault(10);
    AddAttr<int>("batch_size", "the instance number the batch").SetDefault(10);
    AddAttr<int>("hidden_size", "hidden size of the LSTM").SetDefault(100);
    AddAttr<int>("num_layers", "the total layer number of the LSTM")
        .SetDefault(1);
    AddAttr<bool>("is_test", "True if in test phase.").SetDefault(false);
    AddAttr<bool>("fix_seed", "True if it fix dropout seed").SetDefault(false);
    AddAttr<int>("seed", "seed to used if fix_seed is True").SetDefault(0);
    AddComment(R"DOC(
CUDNN LSTM implementation

A four-gate Long Short-Term Memory network with no peephole connections.
In the forward pass the output ht and cell output ct for a given iteration can be computed from the recurrent input ht-1, 
the cell input ct-1 and the previous layer input xt given matrices W, R and biases bW, bR from the following equations:

it = σ(Wi X xt + Ri X ht-1 + bWi + bRi)
ft = σ(Wf X xt + Rf X ht-1 + bWf + bRf)
ot = σ(Wo X xt + Ro X ht-1 + bWo + bRo)
c't = tanh(Wc X xt + Rc X ht-1 + bWc + bRc)
ct = ft * ct-1 + it * c't
ht = ot * tanh(ct)

Where σ is the sigmoid operator: σ(x) = 1 / (1 + e^-x), * represents a point-wise multiplication, 
X represensts a matrix multiplication
and tanh is the hyperbolic tangent function. it, ft, ot, c't represent the input, forget, output and new gates respectively.


)DOC");
  }
};

class CudnnLSTMGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"),
                   "Input(Input) of LSTM should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("W"), "Input(W) of LSTM should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("last_h"),
                   "Input(last_h) of LSTM should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("last_c"),
                   "Input(last_c) of LSTM should not be null.");

    PADDLE_ENFORCE(ctx->HasInput("Cache"),
                   "Input(last_c) of LSTM should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("InitH"),
                   "Input(init_h) of LSTM should not be null.");

    PADDLE_ENFORCE(ctx->HasInput("InitC"),
                   "Input(init_c) of LSTM should not be null.");

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
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(cudnn_lstm, ops::CudnnLSTMOp, ops::CudnnLSTMOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(cudnn_lstm_grad, ops::CudnnLSTMGradOp);

REGISTER_OP_CPU_KERNEL(
    cudnn_lstm,
    ops::CudnnLSTMKernel<paddle::platform::CPUDeviceContext, float>);

REGISTER_OP_CPU_KERNEL(
    cudnn_lstm_grad,
    ops::CudnnLSTMGradKernel<paddle::platform::CPUDeviceContext, float>);
