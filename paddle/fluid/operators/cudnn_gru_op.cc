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
    PADDLE_ENFORCE_EQ(ctx->HasInput("Input"), true,
                      "Input(Input) of GRU should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("W"), true,
                      "Input(Weight) of GRU should not be null.");

    PADDLE_ENFORCE_EQ(ctx->HasInput("InitH"), true,
                      "Input(init_h) of GRU should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("Cache"), true,
                      "Input(Cache) of GRU should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      "Output(Out) of GRU should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasOutput("last_h"), true,
                      "Output(last_h) of GRU should not be null.");

    auto in_dims = ctx->GetInputDim("Input");
    PADDLE_ENFORCE_EQ(in_dims.size(), 3, "Input(X)'s rank must be 3.");

    auto out_dims = in_dims;
    auto hidden_size = ctx->Attrs().Get<int>("hidden_size");
    auto is_bidirec = ctx->Attrs().Get<bool>("is_bidirec");
    const auto num_directions = is_bidirec ? 2 : 1;
    out_dims[2] = hidden_size * num_directions;

    ctx->SetOutputDim("Out", out_dims);
    ctx->SetOutputDim("last_h", ctx->GetInputDim("InitH"));
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
    AddOutput("last_h",
              "(Tensor) the hidden state of the last step. "
              "The shape is ( num_layers x batch_size x hidden_size) if "
              "is_bidirec is False"
              "and When is_bidirec is True, the shape will be (num_layers*2 x "
              "batch_size x hidden_size)");
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
    AddAttr<int>("input_size", "input size ot the Input Tensor").SetDefault(32);
    AddAttr<int>("hidden_size", "hidden size of the GRU").SetDefault(64);
    AddAttr<int>("num_layers", "the total layer number of the GRU")
        .SetDefault(1);
    AddAttr<bool>("is_test", "True if in test phase.").SetDefault(false);
    AddAttr<int>("seed", "seed to used if fix_seed is True").SetDefault(-1);
    AddComment(R"DOC(
CUDNN GRU implements a three-gate recurrent network with cudnn.

$$
update\_gate: u_t = sigmoid(W_{ux}x_{t} + W_{uh}h_{t-1} + bx_u + bh_u) \\
reset\_gate: r_t = sigmoid(W_{rx}x_{t} + W_{rh}h_{t-1} + bx_r + bh_r)  \\
output\_candidate: \\tilde{h_t} = tanh(W_{cx}x_{t} + W_{ch} * dot(r_t, (W_{ch}h_{t-1} + bx_c)) + bh_c) \\
output: h_t = dot(u_t, h_{t-1}) + dot((1 - u_t), \\tilde{h_t})
$$

- W terms denote weight matrices (e.g. $W_{ix}$ is the matrix
  of weights from the input gate to the input)
- The b terms denote bias vectors ($bx_i$ and $bh_i$ are the input gate bias vector).
- sigmoid is the logistic sigmoid function.
- $u, r$ and $c$ are the update gate, reset gate.
  and cell activation vectors, respectively, all of which have the same size as
  the cell output activation vector $h$.
- The $\odot$ is the element-wise product of the vectors.
- `tanh` is the activation functions.
- $\tilde{h_t}$ is also called candidate hidden state,
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
    PADDLE_ENFORCE_EQ(ctx->HasInput("Input"), true,
                      "Input(Input) of GRU should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("W"), true,
                      "Input(W) of GRU should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("last_h"), true,
                      "Input(last_h) of GRU should not be null.");

    PADDLE_ENFORCE_EQ(ctx->HasInput("Cache"), true,
                      "Input(last_c) of GRU should not be null.");
    PADDLE_ENFORCE_EQ(ctx->HasInput("InitH"), true,
                      "Input(init_h) of GRU should not be null.");

    auto SetOutGradDim = [&ctx](const std::string& name) {
      auto g_name = framework::GradVarName(name);
      if (ctx->HasOutput(g_name)) {
        ctx->SetOutputDim(g_name, ctx->GetInputDim(name));
      }
    };

    SetOutGradDim("Input");
    SetOutGradDim("W");
    SetOutGradDim("InitH");
  }
};

template <typename T>
class NotImpleKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_THROW("CPU is not support for this kernel now.");
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
