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
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using framework::Tensor;

template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

class RowConvOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "row_conv");
    OP_INOUT_CHECK(ctx->HasInput("Filter"), "Input", "Filter", "row_conv");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "row_conv");

    auto x_dims = ctx->GetInputDim("X");
    auto filter_dims = ctx->GetInputDim("Filter");
    PADDLE_ENFORCE_EQ(filter_dims.size(),
                      2,
                      platform::errors::InvalidArgument(
                          "Input(Filter)'s dimensions should be 2. Received: "
                          "Input(Filter)'s shape: [%s].",
                          filter_dims));

    ctx->SetOutputDim("Out", x_dims);
    ctx->ShareLoD("X", "Out");
  }
};

class RowConvGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Filter"), "Input", "Filter", "row_conv_grad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   framework::GradVarName("Out"),
                   "row_conv_grad");

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
             "the input(X) is a LodTensor or tensor, LodTensor(X) supports "
             "variable time-length input sequences. The underlying tensor "
             "in this LoDTensor is a matrix with shape (T x N), where T "
             "is the total time steps in this mini-batch and N is the input "
             "data dimension. the shape of Tensor input(X) has shape "
             "(B x T x N), B is batch size;");
    AddInput("Filter",
             "the input(Filter) is a learnable parameter. It "
             "is a 2-D tensor with shape (future_context x N), where, "
             "future_context is the future context length and N is the data "
             "dimension.");
    AddOutput("Out",
              "the output(Out) is a LodTensor or Tensor, which has same type"
              " and same shape as X.");
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
class RowConvKernel<phi::CPUContext, T> : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *x = context.Input<LoDTensor>("X");
    auto *filter = context.Input<Tensor>("Filter");
    auto *out = context.Output<LoDTensor>("Out");

    out->mutable_data<T>(context.GetPlace());

    bool is_tensor = x->lod().empty();
    int batch_size = 0;
    if (is_tensor) {
      batch_size = x->dims()[0];
    } else {
      batch_size = x->lod()[0].size() - 1;
    }
    framework::Vector<size_t> batch_indices(batch_size + 1);
    int input_dim = 0;
    int timesteps = 0;
    if (is_tensor) {
      for (int i = 0; i < batch_size + 1; i++) {
        batch_indices[i] = i;
      }
      input_dim = x->dims()[2];
      timesteps = x->dims()[1];
    } else {
      batch_indices = x->lod()[0];
      input_dim = x->dims()[1];
    }
    size_t num_sequence = batch_indices.size() - 1;

    auto future_context = filter->dims()[0];
    auto weights = EigenMatrix<T>::From(*filter);

    for (size_t i = 0; i < num_sequence; i++) {
      int start = static_cast<int>(batch_indices[i]);
      int end = static_cast<int>(batch_indices[i + 1]);
      int current_timesteps = 0;
      if (is_tensor) {
        current_timesteps = timesteps;
      } else {
        current_timesteps = end - start;
      }
      // int current_timesteps = end - start;
      Tensor cur_input_sequence =
          x->Slice(start, end);  // Current input sequence
      cur_input_sequence =
          cur_input_sequence.Resize({current_timesteps, input_dim});

      Tensor cur_output_sequence =
          out->Slice(start, end);  // Current output sequence
      cur_output_sequence =
          cur_output_sequence.Resize({current_timesteps, input_dim});

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
class RowConvGradKernel<phi::CPUContext, T> : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *x = context.Input<LoDTensor>("X");
    auto *filter = context.Input<Tensor>("Filter");
    auto *d_out = context.Input<LoDTensor>(framework::GradVarName("Out"));
    auto *dx = context.Output<LoDTensor>(framework::GradVarName("X"));
    auto *d_filter = context.Output<Tensor>(framework::GradVarName("Filter"));

    auto &x_lod = x->lod();
    bool is_tensor = x_lod.empty();
    int batch_size = 0;
    if (is_tensor) {
      batch_size = x->dims()[0];
    } else {
      batch_size = x->lod()[0].size() - 1;
    }
    framework::Vector<size_t> batch_indices(batch_size + 1);
    int timesteps = 0;
    int input_dim = 0;
    if (is_tensor) {
      for (int i = 0; i < batch_size + 1; i++) {
        batch_indices[i] = i;
      }
      input_dim = x->dims()[2];
      timesteps = x->dims()[1];
    } else {
      batch_indices = x->lod()[0];
      input_dim = x->dims()[1];
    }

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

        int current_timesteps = 0;
        if (is_tensor) {
          current_timesteps = timesteps;
        } else {
          current_timesteps = end - start;
        }
        Tensor cur_input = x->Slice(start, end);  // Current input sequence
        cur_input = cur_input.Resize({current_timesteps, input_dim});
        Tensor cur_doutput =
            d_out->Slice(start, end);  // Current output grad sequence
        cur_doutput = cur_doutput.Resize({current_timesteps, input_dim});
        auto cur_ip = EigenMatrix<T>::From(cur_input);
        auto cur_dout = EigenMatrix<T>::From(cur_doutput);
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

        int current_timesteps = 0;
        if (is_tensor) {
          current_timesteps = timesteps;
        } else {
          current_timesteps = end - start;
        }

        Tensor cur_doutput =
            d_out->Slice(start, end);  // Current output grad sequence
        cur_doutput = cur_doutput.Resize({current_timesteps, input_dim});
        Tensor cur_dinput =
            dx->Slice(start, end);  // Current input grad sequence
        cur_dinput = cur_dinput.Resize({current_timesteps, input_dim});

        auto cur_dout = EigenMatrix<T>::From(cur_doutput);
        auto cur_dip = EigenMatrix<T>::From(cur_dinput);
        cur_dip.setZero();

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

template <typename T>
class RowConvGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("row_conv_grad");
    op->SetAttrMap(this->Attrs());
    op->SetInput("X", this->Input("X"));
    op->SetInput("Filter", this->Input("Filter"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetOutput(framework::GradVarName("Filter"), this->InputGrad("Filter"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(row_conv,
                  ops::RowConvOp,
                  ops::RowConvOpMaker,
                  ops::RowConvGradOpMaker<paddle::framework::OpDesc>,
                  ops::RowConvGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(row_conv_grad, ops::RowConvGradOp);
REGISTER_OP_CPU_KERNEL(row_conv, ops::RowConvKernel<phi::CPUContext, float>);
REGISTER_OP_CPU_KERNEL(row_conv_grad,
                       ops::RowConvGradKernel<phi::CPUContext, float>);
