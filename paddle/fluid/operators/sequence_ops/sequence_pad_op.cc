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

#include "paddle/fluid/operators/sequence_ops/sequence_pad_op.h"

#include <memory>
#include <string>

namespace paddle {
namespace operators {

class SequencePadOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"),
                      true,
                      platform::errors::NotFound(
                          "Input(X) of SequencePadOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput("PadValue"),
        true,
        platform::errors::NotFound(
            "Input(PadValue) of SequencePadOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"),
                      true,
                      platform::errors::NotFound(
                          "Output(Out) of SequencePadOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasOutput("Length"),
        true,
        platform::errors::NotFound(
            "Output(Length) of SequencePadOp should not be null."));

    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_GE(x_dims.size(),
                      2,
                      platform::errors::InvalidArgument(
                          "The rank of SequencePadOp Input(X) can't be less "
                          "than 2. But the rank we received is %d",
                          x_dims.size()));
    auto time_step_dims = phi::slice_ddim(x_dims, 1, x_dims.size());
    auto pad_value_dims = ctx->GetInputDim("PadValue");
    PADDLE_ENFORCE_EQ(
        pad_value_dims == phi::make_ddim({1}) ||
            pad_value_dims == time_step_dims,
        true,
        platform::errors::InvalidArgument(
            "The SequencePadOp Input(PadValue) must be a scalar or a tensor "
            "whose shape equals to time steps in sequences"));

    int out_dim_0 = -1;

    int padded_length = ctx->Attrs().Get<int>("padded_length");
    if (ctx->IsRuntime()) {
      // run time
      framework::Variable* x_var =
          PADDLE_GET(framework::Variable*, ctx->GetInputVarPtrs("X")[0]);
      const auto& x_lod = x_var->Get<LoDTensor>().lod();
      PADDLE_ENFORCE_EQ(x_lod.empty(),
                        false,
                        platform::errors::NotFound(
                            "The SequencePadOp Input(X) must hold lod info."));
      const auto& x_lod_0 = x_lod[0];
      PADDLE_ENFORCE_GE(
          x_lod_0.size(),
          2,
          platform::errors::InvalidArgument(
              "The size of SequencePadOp Input(X)'s lod info can't be less "
              "than 2. But the size we received is %d",
              x_lod_0.size()));
      PADDLE_ENFORCE_EQ(x_dims[0],
                        static_cast<int64_t>(x_lod_0.back()),
                        platform::errors::InvalidArgument(
                            "The SequencePadOp Input(X)'s lod info mismatches "
                            "the actual tensor shape. The 1st dimension of "
                            "Input(X)'s lod info is %d, the 1st dimension of "
                            "actual tensor shape is %d",
                            x_dims[0],
                            static_cast<int64_t>(x_lod_0.back())));

      int seq_num = x_lod_0.size() - 1;
      int max_seq_len = math::MaximumSequenceLength(x_lod_0);
      if (padded_length == -1) {
        padded_length = max_seq_len;
      }
      PADDLE_ENFORCE_GE(
          padded_length,
          max_seq_len,
          platform::errors::InvalidArgument(
              "The SequencePadOp Attr(padded_length) should be greater than or "
              "equal to the "
              "length of the longest original sequence. But the padded_length "
              "we received is %d, the length of the longest original sequence "
              "is %d",
              padded_length,
              max_seq_len));
      out_dim_0 = seq_num;
    } else {
      // compile time
      if (padded_length == -1) {
        padded_length = 1;
      }
      PADDLE_ENFORCE_GT(
          ctx->GetLoDLevel("X"),
          0,
          platform::errors::InvalidArgument(
              "The LoD level of SequencePadOp Input(X) should be "
              "larger than 0. But the LoD level we received is %d",
              ctx->GetLoDLevel("X")));
    }

    std::vector<int> out_dims_vec{out_dim_0, padded_length};
    std::vector<int> len_dims_vec{out_dim_0};
    auto time_step_dims_vec = phi::vectorize<int>(time_step_dims);
    out_dims_vec.insert(out_dims_vec.end(),
                        time_step_dims_vec.begin(),
                        time_step_dims_vec.end());
    ctx->SetOutputDim("Out", phi::make_ddim(out_dims_vec));
    ctx->SetOutputDim("Length", phi::make_ddim(len_dims_vec));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(ctx, "X");
    return framework::OpKernelType(data_type, ctx.device_context());
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
    AddOutput(
        "Length",
        "(LoDTensor) The output vairable, which contains the actual length of "
        "sequences before padding.");
    AddAttr<int>(
        "padded_length",
        "The length of padded sequences. It can be set to -1 or "
        "any positive int. When it is -1, all sequences will be padded up to "
        "the length of the longest one among them; when it a certain positive "
        "value, it must be greater than the length of the longest original "
        "sequence.")
        .SetDefault(-1);
    AddComment(R"DOC(
      Sequence Pad Operator

      This operator pads sequences in a same batch to a consistent length.
      The length is specified by attribute 'padded_length'. New elements,
      whose values are specified by input 'PadValue', will be appended to
      the end of each sequence, to make their final lengths consistent.

      Following are cases to better explain how this works:

      Case 1:

      Given a 1-level LoDTensor input(X):
          X.lod = [[0, 2,       5]]
          X.data = [a, b, c, d, e]
      and Input(PadValue):
          PadValue.data = [0]
      and attribite 'padded_length' = 4,
      then we get LoDTensor:
          Out.data = [[a, b, 0, 0],
                      [c, d, e, 0]]
          Length.data = [2, 3]

      Case 2:

      Given a 1-level LoDTensor input(X):
          X.lod = [[0,               2,                           5]]
          X.data = [[a1, a2], [b1, b2], [c1, c2], [d1, d2], [e1, e2]]
      and Input(PadValue):
          PadValue.data = [0]
      and attribite 'padded_length' = -1, which mean using the length
      of longest input sequence(3 in this case),
      then we get LoDTensor:
          Out.data = [[[a1, a2], [b1, b2], [0, 0]],
                      [[c1, c2], [d1, d2], [e1, e2]]]
          Length.data = [2, 3]

      Case 3:

      Given a 1-level LoDTensor input(X):
          X.lod = [[0,               2,                           5]]
          X.data = [[a1, a2], [b1, b2], [c1, c2], [d1, d2], [e1, e2]]
      and Input(PadValue):
          PadValue.data = [p1, p2]
      and attribite 'padded_length' = -1, which mean using the length
      of longest input sequence(3 in this case),
      then we get LoDTensor:
          Out.data = [[[a1, a2], [b1, b2], [p1, p2]],
                      [[c1, c2], [d1, d2], [e1, e2]]]
          Length.data = [2, 3]

    )DOC");
  }
};

class SequencePadGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"),
                      true,
                      platform::errors::NotFound(
                          "Input(X) of SequencePadGradOp should not be null."));
    PADDLE_ENFORCE_EQ(
        ctx->HasInput(framework::GradVarName("Out")),
        true,
        platform::errors::NotFound(
            "Input(Out@GRAD) of SequencePadGradOp should not be null."));

    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
      ctx->ShareLoD("X", /*->*/ framework::GradVarName("X"));
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    auto data_type = OperatorWithKernel::IndicateVarDataType(
        ctx, framework::GradVarName("Out"));
    return framework::OpKernelType(data_type, ctx.device_context());
  }
};

template <typename T>
class SequencePadGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("sequence_pad_grad");
    op->SetAttrMap(this->Attrs());
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERER(SequencePadGradOpNoNeedBufferVarsInferer,
                                    "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(sequence_pad,
                  ops::SequencePadOp,
                  ops::SequencePadOpMaker,
                  ops::SequencePadGradOpMaker<paddle::framework::OpDesc>,
                  ops::SequencePadGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(sequence_pad_grad,
                  ops::SequencePadGradOp,
                  ops::SequencePadGradOpNoNeedBufferVarsInferer);
REGISTER_OP_CPU_KERNEL(sequence_pad,
                       ops::SequencePadOpKernel<phi::CPUContext, float>,
                       ops::SequencePadOpKernel<phi::CPUContext, double>,
                       ops::SequencePadOpKernel<phi::CPUContext, int>,
                       ops::SequencePadOpKernel<phi::CPUContext, int64_t>);
REGISTER_OP_CPU_KERNEL(sequence_pad_grad,
                       ops::SequencePadGradOpKernel<phi::CPUContext, float>,
                       ops::SequencePadGradOpKernel<phi::CPUContext, double>,
                       ops::SequencePadGradOpKernel<phi::CPUContext, int>,
                       ops::SequencePadGradOpKernel<phi::CPUContext, int64_t>);
