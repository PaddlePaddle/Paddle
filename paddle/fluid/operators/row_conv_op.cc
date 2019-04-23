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

#include "paddle/fluid/operators/row_conv_op.h"
#include <memory>
#include <string>
#include <vector>

#include "paddle/fluid/framework/eigen.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using framework::Tensor;

template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

class RowConvOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of RowConvOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Filter"),
                   "Input(Filter) of RowConvOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of RowConvOp should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    auto filter_dims = ctx->GetInputDim("Filter");
    PADDLE_ENFORCE_EQ(x_dims.size(), 2, "Input(X)'s rank should be 2.");
    PADDLE_ENFORCE_EQ(filter_dims.size(), 2, "Input(Y)'s rank should be 2.");
    if (ctx->IsRuntime() || (x_dims[1] > 0 && filter_dims[1] > 0)) {
      PADDLE_ENFORCE_EQ(
          x_dims[1], filter_dims[1],
          "The 2nd dimension of Input(X) and Input(Filter) should be same.");
    }

    ctx->SetOutputDim("Out", x_dims);
    ctx->ShareLoD("X", "Out");
  }
};

class RowConvGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Filter"),
                   "Input(Filter) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Gradient of output(Out) should not be null.");

    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      auto dout_dims = ctx->GetInputDim(framework::GradVarName("Out"));
      ctx->SetOutputDim(x_grad_name, dout_dims);
    }

    auto filter_grad_name = framework::GradVarName("Filter");
    if (ctx->HasOutput(filter_grad_name)) {
      auto filter_dims = ctx->GetInputDim("Filter");
      ctx->SetOutputDim(filter_grad_name, filter_dims);
    }
  }
};

class RowConvOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "the input(X) is a LodTensor, which supports "
             "variable time-length input sequences. The underlying tensor "
             "in this LoDTensor is a matrix with shape (T x N), where T "
             "is the total time steps in this mini-batch and N is the input "
             "data dimension.");
    AddInput("Filter",
             "the input(Filter) is a learnable parameter. It "
             "is a 2-D tensor with shape (future_context x N), where, "
             "future_context is the future context length and N is the data "
             "dimension.");
    AddOutput("Out",
              "the output(Out) is a LodTensor, which supports "
              "variable time-length input sequences. The underlying tensor "
              "in this LodTensor is a matrix with shape T x N, i.e., the "
              "same shape as X.");
    AddComment(R"DOC(
:strong:`Row-convolution operator`

The row convolution is called lookahead convolution.  This operator was 
introduced in the following paper for DeepSpeech2:
http://www.cs.cmu.edu/~dyogatam/papers/wang+etal.iclrworkshop2016.pdf 

The main motivation is that a bidirectional RNN, useful in DeepSpeech 
like speech models, learns representation for a sequence by performing a 
forward and a backward pass through the entire sequence. However, unlike 
unidirectional RNNs, bidirectional RNNs are challenging to deploy in an online
and low-latency setting. The lookahead convolution incorporates information 
from future subsequences in a computationally efficient manner to improve 
unidirectional recurrent neural networks. The row convolution operator is 
different from the 1D sequence convolution, and is computed as follows:

Given an input sequence $X$ of length $t$ and input dimension $D$, 
and a filter ($W$) of size $context \times D$,
the output sequence is convolved as:

$$
out_{i} = \\sum_{j=i}^{i + context - 1} X_{j} \\cdot W_{j-i}
$$

In the above equation:

* $Out_{i}$: The i-th row of output variable with shape [1, D].

* $context$: Future context size.

* $X_{j}$: The j-th row of input variable with shape [1, D].

* $W_{j-i}$: The (j-i)-th row of parameters with shape [1, D].

More details about row_conv please refer to
the design document
https://github.com/PaddlePaddle/Paddle/issues/2228#issuecomment-303903645 .

)DOC");
  }
};

template <typename T>
class RowConvKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *x = context.Input<LoDTensor>("X");
    auto *filter = context.Input<Tensor>("Filter");
    auto *out = context.Output<LoDTensor>("Out");

    out->mutable_data<T>(context.GetPlace());

    auto batch_indices = x->lod()[0];
    auto input_dim = x->dims()[1];  // 'in' is of size T x N
    size_t num_sequence = batch_indices.size() - 1;

    auto future_context = filter->dims()[0];
    auto weights = EigenMatrix<T>::From(*filter);

    for (size_t i = 0; i < num_sequence; i++) {
      int start = static_cast<int>(batch_indices[i]);
      int end = static_cast<int>(batch_indices[i + 1]);
      int current_timesteps = end - start;
      Tensor cur_input_sequence =
          x->Slice(start, end);  // Current input sequence
      Tensor cur_output_sequence =
          out->Slice(start, end);  // Current output sequence
      auto cip_seq = EigenMatrix<T>::From(cur_input_sequence);
      auto cot_seq = EigenMatrix<T>::From(cur_output_sequence);

      for (int k = 0; k < current_timesteps;
           k++) {  // For different time steps in the same sequence
        for (int w = 0; (w < future_context) && ((k + w) < current_timesteps);
             w++) {
          for (int d = 0; d < input_dim; d++) {
            if (w == 0) {
              cot_seq(k, d) = weights(w, d) * cip_seq(k + w, d);
            } else {
              cot_seq(k, d) += weights(w, d) * cip_seq(k + w, d);
            }
          }
        }
      }
    }
  }
};

template <typename T>
class RowConvGradKernel<platform::CPUDeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *x = context.Input<LoDTensor>("X");
    auto *filter = context.Input<Tensor>("Filter");
    auto *d_out = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto *dx = context.Output<LoDTensor>(framework::GradVarName("X"));
    auto *d_filter = context.Output<Tensor>(framework::GradVarName("Filter"));

    auto input_dim = x->dims()[1];  // 'x' is of size T x N
    auto batch_indices = x->lod()[0];
    size_t num_sequence = batch_indices.size() - 1;
    auto future_context = filter->dims()[0];

    if (d_filter) {
      d_filter->mutable_data<T>(context.GetPlace());
      auto dweights =
          EigenMatrix<T>::From(*d_filter);  // Gradient of weight matrix
      dweights.setZero();

      for (size_t i = 0; i < num_sequence; i++) {  // For different sequences
        int start = static_cast<int>(batch_indices[i]);
        int end = static_cast<int>(batch_indices[i + 1]);

        Tensor cur_input = x->Slice(start, end);  // Current input sequence
        Tensor cur_doutput =
            d_out->Slice(start, end);  // Current output grad sequence

        auto cur_ip = EigenMatrix<T>::From(cur_input);
        auto cur_dout = EigenMatrix<T>::From(cur_doutput);
        int current_timesteps = end - start;

        for (int k = 0; k < current_timesteps;
             k++) {  // For different time steps in the same sequence
          for (int w = 0; (w < future_context) && ((k + w) < current_timesteps);
               w++) {
            // For dweights (Updating the gradient of weight matrix)
            for (int d = 0; d < input_dim; d++) {
              dweights(w, d) += cur_ip(k + w, d) * cur_dout(k, d);
            }
          }
        }
      }
    }

    if (dx) {
      dx->mutable_data<T>(context.GetPlace());
      auto weights = EigenMatrix<T>::From(*filter);
      for (size_t i = 0; i < num_sequence; i++) {  // For different sequences
        int start = static_cast<int>(batch_indices[i]);
        int end = static_cast<int>(batch_indices[i + 1]);

        Tensor cur_doutput =
            d_out->Slice(start, end);  // Current output grad sequence
        Tensor cur_dinput =
            dx->Slice(start, end);  // Current input grad sequence

        auto cur_dout = EigenMatrix<T>::From(cur_doutput);
        auto cur_dip = EigenMatrix<T>::From(cur_dinput);
        cur_dip.setZero();
        int current_timesteps = end - start;

        for (int k = 0; k < current_timesteps;
             k++) {  // For different time steps in the same sequence
          for (int w = 0; (w < future_context) && ((k + w) < current_timesteps);
               w++) {
            // For dinput (Updating the gradient wrt input)
            for (int d = 0; d < input_dim; d++) {
              cur_dip(k + w, d) += weights(w, d) * cur_dout(k, d);
            }
          }
        }
      }
    }
  }
};

class RowConvGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("row_conv_grad");
    op->SetAttrMap(Attrs());
    op->SetInput("X", Input("X"));
    op->SetInput("Filter", Input("Filter"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetOutput(framework::GradVarName("Filter"), InputGrad("Filter"));
    return op;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(row_conv, ops::RowConvOp, ops::RowConvOpMaker,
                  ops::RowConvGradOpDescMaker);
REGISTER_OPERATOR(row_conv_grad, ops::RowConvGradOp);
REGISTER_OP_CPU_KERNEL(
    row_conv, ops::RowConvKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    row_conv_grad,
    ops::RowConvGradKernel<paddle::platform::CPUDeviceContext, float>);
