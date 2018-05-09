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

    auto out_dims = x_dims;

    if (ctx->IsRuntime()) {
      framework::Variable* x_var =
          boost::get<framework::Variable*>(ctx->GetInputVarPtrs("X")[0]);

      auto& x_lod = x_var->Get<LoDTensor>().lod();

      PADDLE_ENFORCE_GE(x_lod.size(), 1,
                        "Input(X) should be sequences containing lod.");

      auto last_level_lod = x_lod[x_lod.size() - 1];
      size_t max_len = 0;

      for (size_t i = 1; i < last_level_lod.size(); ++i) {
        auto seq_len = last_level_lod[i] - last_level_lod[i - 1];
        max_len = max_len < seq_len ? seq_len : max_len;
      }

      out_dims[0] = max_len * (last_level_lod.size() - 1);
    } else {
      framework::VarDesc* x_desc =
          boost::get<framework::VarDesc*>(ctx->GetInputVarPtrs("X")[0]);
      PADDLE_ENFORCE_GE(x_desc->GetLoDLevel(), 1,
                        "Input(X) should be sequences containing lod.");
      out_dims[0] = -1;
    }

    ctx->SetOutputDim("Out", out_dims);
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
    AddAttr<float>("pad_value",
                   "(float, default 0.0) Value to be padded "
                   "to the end of each sequence.");
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
