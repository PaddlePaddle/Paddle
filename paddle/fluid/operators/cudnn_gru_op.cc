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

#include <string>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class CudnnGRUOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"),
                   "Input(Input) of GRU should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("W"),
                   "Input(Weight) of GRU should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("InitH"),
                   "Input(init_h) of GRU should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("last_h"),
                   "Output(last_h) of GRU should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Cache"),
                   "Input(Cache) of GRU should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of GRU should not be null.");

    auto in_dims = ctx->GetInputDim("Input");

    int dims_1 = ctx->Attrs().Get<int>("hidden_size");
    // input is TXM for lodtensor
    if (in_dims.size() == 2) {
      std::vector<int64_t> output_dims = {in_dims[0], dims_1};
      ctx->SetOutputDim("Out", framework::make_ddim(output_dims));
      ctx->ShareLoD("Input", /*->*/ "Out");
    } else {
      // input is cudnn formal data BXSXH
      ctx->SetOutputDim("last_h", ctx->GetInputDim("InitH"));
      ctx->SetOutputDim("Out", ctx->GetInputDim("Input"));
    }
  }
};

class CudnnGRUOpMaker : public framework::OpProtoAndCheckerMaker {
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
             "(Tensor) the initial hidden state of the GRU"
             "input. This is a tensor with shape (num_layers x batch_size x "
             "hidden_size)"
             "and When is_bidirec is True, the shape will be (num_layers*2 x "
             "batch_size x hidden_size)");
    AddOutput("last_h",
              "(Tensor) the hidden state of the last step. "
              "The shape is ( num_layers x batch_size x hidden_size) if "
              "is_bidirec is False"
              "and When is_bidirec is True, the shape will be (num_layers*2 x "
              "batch_size x hidden_size)");
    AddInput("W",
             "(Tensor) the learnable hidden-hidden weights."
             " The shape is (N), where N is total weight size of the GRU. "
             " cudnn concatenate all the weight to one Tensor");
    AddInput("Cache",
             "The cache of dropout op, a RAW type variable including random "
             "number generator states and some descriptors, which is used in "
             "cudnn kernel.")
        .AsDispensable();
    AddOutput("Out",
              "(Tensor) the hidden state of GRU operator. "
              "The shape is ( seq_len x batch_size x hidden_size) if "
              "is_bidirec is False"
              "and When is_bidirec is True, the shape will be ( seq_len x "
              "batch_size x hidden_size * 2) ");
    AddAttr<int>("max_len",
                 "max length of the GRU op"
                 "the first dim of the Input can NOT be greater than max_len")
        .SetDefault(20);
    AddAttr<float>(
        "dropout_prob",
        "dropout prob of the dropout op"
        "the dropout ONLY work between gru layers, not between time steps"
        "There is no dropout work on the Out tensor")
        .SetDefault(0.0);
    AddAttr<bool>("is_bidirec",
                  "is_bidirec"
                  "if it is bidirection rnn"
                  "The will affect the shape of the Out, last_h, and last_c")
        .SetDefault(false);
    AddAttr<int>("input_size", "input size ot the Input Tensor").SetDefault(10);
    AddAttr<int>("hidden_size", "hidden size of the GRU").SetDefault(100);
    AddAttr<int>("num_layers", "the total layer number of the GRU")
        .SetDefault(1);
    AddAttr<bool>("is_test", "True if in test phase.").SetDefault(false);
    AddAttr<int>("seed", "seed to used if fix_seed is True").SetDefault(-1);
    AddComment(R"DOC(
CUDNN GRU implementation

A four-gate Long Short-Term Memory network with no peephole connections.
In the forward pass the output ht and cell output ct for a given iteration can be computed from the recurrent input ht-1, 
the cell input ct-1 and the previous layer input xt given matrices W, R and biases bW, bR from the following equations:
$$
update\_gate: u_t = sigmoid(W_{ux}x_{t} + W_{uh}h_{t-1} + bx_u + bh_u) \\
reset\_gate: r_t = sigmoid(W_{rx}x_{t} + W_{rh}h_{t-1} + bx_r + bh_r)  \\
output\_candidate: {h}_t = tanh(W_{cx}x_{t} + W_{ch} * dot(r_t, (W_{ch}h_{t-1} + bx_c)) + bh_c) \\
output: h_t = dot(u_t, h_{t-1}) + dot((1 - u_t), {h}_t)
$$

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

class CudnnGRUGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Input"),
                   "Input(Input) of GRU should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("W"), "Input(W) of GRU should not be null.");

    PADDLE_ENFORCE(ctx->HasInput("Cache"),
                   "Input(Cache) of GRU should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("InitH"),
                   "Input(init_h) of GRU should not be null.");

    auto SetOutGradDim = [&ctx](const std::string& name) {
      auto g_name = framework::GradVarName(name);
      if (ctx->HasOutput(g_name)) {
        ctx->SetOutputDim(g_name, ctx->GetInputDim(name));
      }
    };

    SetOutGradDim("Input");
    SetOutGradDim("W");
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
REGISTER_OPERATOR(cudnn_gru, ops::CudnnGRUOp, ops::CudnnGRUOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(cudnn_gru_grad, ops::CudnnGRUGradOp);

REGISTER_OP_CPU_KERNEL(cudnn_gru, ops::NotImpleKernel<float>);
REGISTER_OP_CPU_KERNEL(cudnn_gru_grad, ops::NotImpleKernel<float>);
