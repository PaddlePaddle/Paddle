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

#include "paddle/operators/sequence_conv_op.h"

namespace paddle {
namespace operators {

class SequenceConvOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SequenceConvOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Filter"),
                   "Input(Filter) of SequenceConvOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SequenceConvOp should not be null.");
    // PaddingData mast be not empty. Otherwise(EnforceNotMet: enforce numel() >
    // 0 failed, 0 <= 0)
    PADDLE_ENFORCE(ctx->HasInput("PaddingData"),
                   "Input(PaddingData) of SequenceConvOp should not be null.");

    int context_length = ctx->Attrs().Get<int>("context_length");
    bool padding_trainable = ctx->Attrs().Get<bool>("padding_trainable");
    int context_start = ctx->Attrs().Get<int>("context_start");

    auto in_dims = ctx->GetInputDim("X");
    auto filter_dims = ctx->GetInputDim("Filter");
    PADDLE_ENFORCE(in_dims.size() == 2 && filter_dims.size() == 2,
                   "Input(X, Filter) should be 2-D tensor.");
    PADDLE_ENFORCE(
        filter_dims[0] == context_length && filter_dims[1] == in_dims[1],
        "Filter's shape should be (context_length x "
        "number_of_input_features).");

    if (padding_trainable) {
      framework::DDim padding_dim = ctx->GetInputDim("PaddingData");
      int up_pad = std::max(0, -context_start);
      int down_pad = std::max(0, context_start + context_length - 1);
      int total_pad = up_pad + down_pad;
      int input_width = static_cast<int>(in_dims[1]);

      if (context_start == 0 && context_length == 1) {
        PADDLE_THROW(
            "If context_start is 0 and context_length is 1, padding_trainable "
            "should be false.");
      }
      PADDLE_ENFORCE(padding_dim.size() == 2,
                     "Input(PaddingData) should be 2-D tensor.");
      PADDLE_ENFORCE(
          padding_dim[0] == total_pad && padding_dim[1] == input_width,
          "Input(PaddingData)'s shape is not consistent with 'context_start' "
          "and 'context_length'.");
    }

    in_dims[1] = 1;
    ctx->SetOutputDim("Out", in_dims);
  }
};

class SequenceConvGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Gradient of output(Out) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("X"), "The input(X) should not be null.");

    if (ctx->Attrs().Get<bool>("padding_trainable") &&
        ctx->HasOutput(framework::GradVarName("PaddingData"))) {
      ctx->SetOutputDim(framework::GradVarName("PaddingData"),
                        ctx->GetInputDim("PaddingData"));
    }
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
    }
    if (ctx->HasOutput(framework::GradVarName("Filter"))) {
      ctx->SetOutputDim(framework::GradVarName("Filter"),
                        ctx->GetInputDim("Filter"));
    }
  }
};

class SequenceConvOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SequenceConvOpMaker(framework::OpProto* proto,
                      framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "(A float LoDTensor) the input of SequenceConvOp, a vector of "
             "2-D matrix of size (minibatch, number_of_input_features).");
    AddInput("PaddingData",
             "(A float LoDTensor) the input of SequenceConvOp, a vector of "
             "2-D matrix of size (up_pad + down_pad, "
             "number_of_input_features). ");
    AddInput("Filter",
             "(A float LoDTensor) the input of SequenceConvOp, a vector of "
             "2-D matrix of size (context_length x number_of_input_features).");
    AddOutput("Out",
              "(A float LoDTensor) the output of SequenceConvOp, a vector "
              "of 2-D matrix of size (minibatch, 1).");

    AddAttr<bool>("padding_trainable",
                  "(bool, default false) the padding data of SequenceConvOp "
                  "is trainable or not.")
        .SetDefault(false);
    AddAttr<int>("context_length",
                 "(int, default 3) the context_length of SequenceConvOp.")
        .SetDefault(3)
        .GreaterThan(0);
    AddAttr<int>("context_start",
                 "(int, default 0) the context_start of SequenceConvOp.")
        .SetDefault(0);
    AddAttr<int>("context_stride",
                 "(int, default 1) the context_stride of SequenceConvOp. "
                 "Currently, sequence_project_op only support "
                 "context_stride=1.")
        .SetDefault(1)
        .GreaterThan(0);

    AddComment(R"DOC(
    SequenceConvOp performs convolution operation on features of
    context_length time-steps of each instance.
    The convolution operation calculates the output based on the input, filter
    and strides, paddings parameters. The size of each dimension of the
    parameters is checked in the infer-shape.

Example:
  Input:
       X shape: (minibatch, number_of_input_features)
       Filter shape: (context_length, number_of_input_features)
  Output:
       Out shape: (minibatch, 1)
    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(sequence_conv, ops::SequenceConvOp, ops::SequenceConvOpMaker,
            sequence_conv_grad, ops::SequenceConvGradOp);

REGISTER_OP_CPU_KERNEL(
    sequence_conv, ops::SequenceConvKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    sequence_conv_grad,
    ops::SequenceConvGradKernel<paddle::platform::CPUPlace, float>);
