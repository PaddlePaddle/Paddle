/* Copyright (c) 2017 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/operators/row_conv_op.h"
#include "paddle/framework/eigen.h"

namespace paddle {
namespace operators {

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
    PADDLE_ENFORCE_EQ(
        x_dims[1], filter_dims[1],
        "The 2nd dimension of Input(X) and Input(Filter) should be same.");
    ctx->SetOutputDim("Out", x_dims);
    ctx->ShareLoD("X", "Out");
  }
};

class RowConvGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Filter"),
                   "Input(Filter) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Gradient of output(Out) should not be null.");

    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      auto x_dims = ctx->GetInputDim("X");
      ctx->SetOutputDim(x_grad_name, x_dims);
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
  RowConvOpMaker(framework::OpProto *proto,
                 framework::OpAttrChecker *op_checker)
      : framework::OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "(LoDTensor), the input(X) is a LodTensor, which supports "
             "variable time-length input sequences. The underlying tensor "
             "in this LoDTensor is a matrix with shape (T x N), where T "
             "is the total time steps in this mini-batch and N is the input "
             "data dimension.");
    AddInput("Filter",
             "(Tensor), the input(Filter) is a learnable parameter. It "
             "is a 2-D tensor with shape (future_context x N), where, "
             "future_context is the batch size and N is the data dimension.");
    AddOutput("Out",
              "(LoDTensor), the output(Out) is a LodTensor, which supports "
              "variable time-length input sequences. The underlying tensor "
              "in this LodTensor is a matrix with shape T x N, i.e., the "
              "same shape as X.");
    AddComment(R"DOC(
Row-convolution Operator.

Give paper reference: 

The equation is:

$$Out[i] = \sum_{j=-(N-1)/2}^{(N-1)/2} X_{i+j} * Y_{j}$$

where X's index is computed modulo M, and Y's index is computed modulo N.

Both inputs X and Y can carry LoD (Level of Details) information.
However, the output only shares the LoD information with input X.

)DOC");
  }
};

template <typename T>
class RowConvKernel<platform::CPUPlace, T> : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *in = context.Input<LoDTensor>("X");
    auto *filter = context.Input<Tensor>("Filter");
    auto *out = context.Output<LoDTensor>("Out");

    out->mutable_data<T>(context.GetPlace());
    context.ShareLoD("X", "Out");

    auto batch_indices = in->lod()[0];
    auto input_dim = in->dims()[1];  // 'in' is of size T x N
    size_t num_sequences = batch_indices.size() - 1;

    auto context_length = filter->dims()[0];
    auto weights = EigenMatrix<T>::From(*filter);

    for (size_t i = 0; i < num_sequences; i++) {
      int start = static_cast<int>(batch_indices[i]);
      int end = static_cast<int>(batch_indices[i + 1]);
      int current_timesteps = end - start + 1;
      Tensor cur_input_sequence =
          in->Slice(start, end);  // Current input sequence
      Tensor cur_output_sequence =
          out->Slice(start, end);  // Current output sequence
      auto cip_seq = EigenMatrix<T>::From(cur_input_seqeunce);
      auto cot_seq = EigenMatrix<T>::From(cur_output_sequence);

      for (size_t k = 0; k < current_timesteps;
           k++) {  // For different time steps in the same sequence
        for (size_t w = 0;
             (w < context_length) && ((k + w) < current_timesteps); w++) {
          for (size_t d = 0; d < input_dim; d++) {
            if (w == 0) {
              cot_seq(k, d) = weights(w, d) * cip_seq(k + w, d);
            } else {
              cot_seq(k, d) += weights(w, d) * cip_seq(k + w, d);
            }
          }
        }
      }
    }
  };

  template <typename T>
  class RowConvGradKernel<platform::CPUPlace, T>
      : public framework::OpKernel<T> {
   public:
    void Compute(const framework::ExecutionContext &context) const override {
      auto *in = context.Input<LoDTensor>("X");
      auto *filter = context.Input<Tensor>("Filter");
      auto *dOut = context.Input<LoDTensor>(framework::GradVarName("Out"));
      auto *din = context.Output<LoDTensor>(framework::GradVarName("X"));
      auto *dfilter = context.Output<Tensor>(framework::GradVarName("Filter"));

      if (din) din->mutable_data<T>(context.GetPlace());
      if (dfilter) auto batch_indices = in->lod()[0];
      auto input_dim = in->dims()[1];  // 'in' is of size T x N
      size_t num_sequences = batch_indices.size() - 1;

      auto context_length = filter->dims()[0];

      if (dfilter) {
        dfilter->mutable_data<T>(context.GetPlace());
        auto dweights =
            EigenMatrix<T>::From(*dfilter);  // Gradient of weight matrix
        dweights.setZero();

        for (size_t i = 0; i < num_sequences; i++) {  // For different sequences
          int start = static_cast<int>(batch_indices[i]);
          int end = static_cast<int>(batch_indices[i + 1]);

          Tensor cur_input = in->Slice(start, end);  // Current input sequence
          Tensor cur_doutput =
              dOut->Slice(start, end);  // Current output grad sequence

          auto cur_ip = EigenMatrix<T>::From(cur_input);
          auto cur_dout = EigenMatrix<T>::From(cur_doutput);
          int current_timesteps = end - start + 1;

          for (size_t k = 0; k < current_timesteps;
               k++) {  // For different time steps in the same sequence
            for (size_t w = 0;
                 (w < context_length) && ((k + w) < current_timesteps); w++) {
              int start = k + w;
              // For dweights (Updating the gradient of weight matrix)
              for (size_t d = 0; d < input_dim; d++) {
                dweights(w, d) += cip_seq(start, d) * cur_dout(k, d);
              }
            }
          }
        }
      }

      if (din) {
        auto weights = EigenMatrix<T>::From(*filter);
        for (size_t i = 0; i < num_sequences; i++) {  // For different sequences
          int start = static_cast<int>(batch_indices[i]);
          int end = static_cast<int>(batch_indices[i + 1]);

          Tensor cur_doutput =
              dOut->Slice(start, end);  // Current output grad sequence
          Tensor cur_dinput =
              din->Slice(start, end);  // Current input grad sequence

          auto cur_dout = EigenMatrix<T>::From(cur_doutput);
          auto cur_dip = EigenMatrix<T>::From(cur_dinput);
          cur_dip.setZero();
          int current_timesteps = end - start + 1;

          for (size_t k = 0; k < current_timesteps;
               k++) {  // For different time steps in the same sequence
            for (size_t w = 0;
                 (w < context_length) && ((k + w) < current_timesteps); w++) {
              int start = k + w;
              // For dinput (Updating the gradient wrt input)
              for (size_t d = 0; d < input_dim; d++) {
                cur_dip(start, d) += weights(w, d) * cur_dout(k, d);
              }
            }
          }
        }
      }
    }
  };
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(row_conv, ops::RowConvOp, ops::RowConvOpMaker, row_conv_grad,
            ops::RowConvGradOp);
REGISTER_OP_CPU_KERNEL(row_conv,
                       ops::RowConvKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    row_conv_grad, ops::RowConvGradKernel<paddle::platform::CPUPlace, float>);
