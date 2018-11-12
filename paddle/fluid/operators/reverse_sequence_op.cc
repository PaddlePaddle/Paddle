// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/reverse_sequence_op.h"

namespace paddle {
namespace operators {

class ReverseSequenceOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be empty.");
    PADDLE_ENFORCE(ctx->HasInput("SeqLen"),
                   "Input(SeqLen) should not be empty.");
    PADDLE_ENFORCE(ctx->HasOutput("Y"), "Output(Y) should not be empty.");

    auto x_dim = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(
        x_dim.size(), 3,
        "Input(X) must be 3D tensor(seq_lengths, batch, embeddings).");

    PADDLE_ENFORCE_EQ(ctx->GetInputDim("SeqLen").size(), 1,
                      "Input(SeqLen) must be 1D tensor.");
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("SeqLen")[0], x_dim[1],
                      "Input(SeqLen)'s size must be equal to batch_size.");

    ctx->SetOutputDim("Y", x_dim);
    ctx->ShareLoD("X", "Y");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    auto input_data_type =
        framework::ToDataType(ctx.Input<framework::LoDTensor>("X")->type());
    return framework::OpKernelType(input_data_type, ctx.GetPlace());
  }
};

class ReverseSequenceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input LoDTensor of sequence_reverse op, and it must be 3D "
             "tensor(seq_lengths, batch, embeddings).");
    AddInput("SeqLen",
             "The input LoDTensor of sequence_reverse op, it must be 3D "
             "tensor(batch).");
    AddOutput("Y",
              "The output LoDTensor of sequence_reverse op, it is 3D "
              "tensor(seq_lengths, batch, embeddings).");
    AddComment(R"DOC(
ReverseSequence Operator.

    )DOC");
  }
};

class ReverseSequenceGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("reverse_sequence");
    op->SetInput("X", OutputGrad("Y"));
    op->SetInput("SeqLen", Input("SeqLen"));
    op->SetOutput("Y", InputGrad("X"));
    op->SetAttrMap(Attrs());
    return op;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(reverse_sequence, ops::ReverseSequenceOp,
                  ops::ReverseSequenceOpMaker,
                  ops::ReverseSequenceGradOpDescMaker);

REGISTER_OP_CPU_KERNEL(
    reverse_sequence,
    ops::ReverseSequenceOpKernel<paddle::platform::CPUDeviceContext, int>,
    ops::ReverseSequenceOpKernel<paddle::platform::CPUDeviceContext, int64_t>,
    ops::ReverseSequenceOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::ReverseSequenceOpKernel<paddle::platform::CPUDeviceContext, double>);
