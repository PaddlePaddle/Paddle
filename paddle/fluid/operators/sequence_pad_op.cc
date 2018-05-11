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

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of SequencePadOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of SequencePadOp should not be null.");

    auto x_dims = ctx->GetInputDim("X");

    PADDLE_ENFORCE_EQ(x_dims.size(), 2,
                      "Only support 2-D tensor, rank of Input(X) should be 2.");

    int lod_level = ctx->Attrs().Get<int>("lod_level");

    int64_t max_len = -1;
    int64_t seq_num = -1;
    int x_lod_size = -1;

    if (ctx->IsRuntime()) {
      framework::Variable* x_var =
          boost::get<framework::Variable*>(ctx->GetInputVarPtrs("X")[0]);

      auto& x_lod = x_var->Get<LoDTensor>().lod();

      x_lod_size = x_lod.size();

      auto x_abs_offset = framework::ToAbsOffset(x_lod)[lod_level];

      PADDLE_ENFORCE_EQ(x_dims[0], static_cast<int64_t>(x_abs_offset.back()),
                        "The first dimension of `X` should be equal to sum "
                        "of all sequences' length.");

      seq_num = x_abs_offset.size() - 1;

      for (size_t i = 1; i <= seq_num; ++i) {
        int64_t seq_len = x_abs_offset[i] - x_abs_offset[i - 1];
        max_len = max_len < seq_len ? seq_len : max_len;
      }
    } else {
      framework::VarDesc* x_desc =
          boost::get<framework::VarDesc*>(ctx->GetInputVarPtrs("X")[0]);
      x_lod_size = x_desc->GetLoDLevel();
    }

    PADDLE_ENFORCE(lod_level >= 0 && lod_level < x_lod_size,
                   "Invalid `lod_level` which should be at least 0 and less "
                   "than maximum lod level of `X`");

    ctx->SetOutputDim("Out", {seq_num, max_len, x_dims[1]});
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::LoDTensor>("X")->type()),
        ctx.device_context());
  }
};

class SequencePadOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SequencePadOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "(LoDTensor, default LoDTensor<float>) Input variable which "
             "should contain lod information. Length of each sequence would "
             "be computed from the most bottom level lod.");
    AddOutput("Out",
              "(Tensor) Output variable which would be a common tensor "
              "without lod. Each sequence would be padded to the maximum "
              "length.");
    AddAttr<float>("lod_level",
                   "(int, default 0) Specify which level lod to referred to.");
    AddAttr<float>("pad_value",
                   "(float, default 0.0) Specify which value to be padded to "
                   "the end of each sequence.");
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
