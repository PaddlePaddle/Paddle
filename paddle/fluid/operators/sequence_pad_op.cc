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

#include "paddle/fluid/operators/sequence_pad_op.h"

namespace paddle {
namespace operators {

class SequencePadOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SequencePadOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("PadValue"),
                   "Input(PadValue) of SequencePadOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SequencePadOp should not be null.");

    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_GE(x_dims.size(), 2,
                      "The rank of Input(x) can't be less than 2.");
    auto time_step_dims = framework::slice_ddim(x_dims, 1, x_dims.size());
    auto pad_value_dims = ctx->GetInputDim("PadValue");
    PADDLE_ENFORCE(pad_value_dims == framework::make_ddim({1}) ||
                       pad_value_dims == time_step_dims,
                   "The Input(PadValue) must be a scalar or a tensor whose "
                   "shape equals to time steps in sequences");

    int batch_dim_size = -1;

    if (ctx->IsRuntime()) {
      // run time
      framework::Variable* x_var =
          boost::get<framework::Variable*>(ctx->GetInputVarPtrs("X")[0]);
      const auto& x_lod = x_var->Get<LoDTensor>().lod();
      PADDLE_ENFORCE(!x_lod.empty(), "The Input(X) must hold lod info.");
      const auto& x_lod_0 = x_lod[0];
      PADDLE_ENFORCE_GE(x_lod_0.size(), 2,
                        "The Input(X)'s lod info is corrupted.");
      PADDLE_ENFORCE_EQ(
          x_dims[0], static_cast<int64_t>(x_lod_0.back()),
          "The Input(X)'s lod info mismatches the actual tensor shape.");

      int seq_num = x_lod_0.size() - 1;
      int max_seq_len = math::MaximumSequenceLength(x_lod_0);
      int padded_length = ctx->Attrs().Get<int>("padded_length");
      if (padded_length == -1) {
        padded_length = max_seq_len;
      }
      PADDLE_ENFORCE_GE(padded_length, max_seq_len,
                        "The Attr(padded_length) must be -1 or an int greater "
                        "than the length of the longest original sequence.");
      batch_dim_size = padded_length * seq_num;
    } else {
      // compile time
      framework::VarDesc* x_desc =
          boost::get<framework::VarDesc*>(ctx->GetInputVarPtrs("X")[0]);
      PADDLE_ENFORCE_GE(x_desc->GetLoDLevel(), 1);
    }

    auto out_dims = x_dims;
    out_dims[0] = batch_dim_size;
    ctx->SetOutputDim("Out", out_dims);
  }
};

class SequencePadOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(LoDTensor, default LoDTensor<float>) Input variable which "
             "should contain lod information.");
    AddInput("PadValue",
             "(LoDTensor), this Tensor holds values that will be fill into "
             "padded steps. It can be a scalar or a tensor whose shape equals "
             "to time steps in sequences. If it's a scalar, it will be "
             "automatically broadcasted to the shape of time step.");
    AddOutput(
        "Out",
        "(LoDTensor) The output vairable, which contains padded sequences.");
    AddAttr<int>(
        "padded_length",
        "The length of padded sequences. It can be setted to -1 or "
        "any positive int. When it is -1, all sequences will be padded up to "
        "the length of the longest one among them; when it a certain positive "
        "value, it must be greater than the length of the longest original "
        "sequence.")
        .SetDefault(-1);
    AddComment(R"DOC(

    )DOC");
  }
};

class SequencePadGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SequencePadGradOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) of SequencePadGradOp should not be null.");

    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
      ctx->ShareLoD("X", /*->*/ framework::GradVarName("X"));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sequence_pad, ops::SequencePadOp, ops::SequencePadOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(sequence_pad_grad, ops::SequencePadGradOp);
REGISTER_OP_CPU_KERNEL(
    sequence_pad,
    ops::SequencePadOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SequencePadOpKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SequencePadOpKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SequencePadOpKernel<paddle::platform::CPUDeviceContext, int64_t>);
REGISTER_OP_CPU_KERNEL(
    sequence_pad_grad,
    ops::SequencePadGradOpKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SequencePadGradOpKernel<paddle::platform::CPUDeviceContext, double>,
    ops::SequencePadGradOpKernel<paddle::platform::CPUDeviceContext, int>,
    ops::SequencePadGradOpKernel<paddle::platform::CPUDeviceContext, int64_t>);
