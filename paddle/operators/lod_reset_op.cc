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

#include "paddle/operators/lod_reset_op.h"

namespace paddle {
namespace operators {

class LoDResetOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    // input check
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of LoDResetOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of LoDResetOp should not be null.");
    // If target LoD is not set form Input(), then it must be set from Attr().
    if (!ctx->HasInput("TargetLoD")) {
      auto level0 = ctx->Attrs().Get<std::vector<int>>("target_lod");
      PADDLE_ENFORCE(level0.size() > 1,
                     "Target LoD is not found, should be set to be a valid one "
                     "through Input() or Attr().");
    }
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::LoDTensor>("X")->type()),
        ctx.device_context());
  }
};

class LoDResetOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LoDResetOpMaker(framework::OpProto *proto,
                  framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(LoDTensor) The input tensor of lod_reset operator.");
    AddInput("TargetLoD",
             "(Tensor, optional) The target level 0 LoD from Input().")
        .AsDispensable();
    AddOutput("Out", "(LoDTensor) The output tensor of lod_reset operator.");
    AddAttr<std::vector<int>>("target_lod",
                              "The target level 0 LoD from Attr().")
        .SetDefault(std::vector<int>{});
    AddComment(R"DOC(LoDReset operator

Reset LoD of Input(X) into a new one specified by Input(TargetLoD) or
Attr(target_lod), or set LoD for Input(X) if it doesn't have one.
Currently the lod_reset operator only supports the reset of level 0 LoD.
At least one of Input(TargetLoD) and Attr(target_lod) must be set,
and if both of them are set, Input(TargetLoD) will be chosen as the
target LoD.

An example:
Given a float LoDTensor X with shape (6, 1), its transpose form represents

    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],

with LoD = [[0, 2, 5, 6]] and the three (transposed) sequences look like

    [1.0, 2.0], [3.0, 4.0, 5.0], [6.0].

If target LoD = [0, 4, 6], the lod_reset operator will reset the LoD and
the sequences that the LoDTensor Output(Out) contains becomes:

    [1.0, 2.0, 3.0, 4.0], [5.0, 6.0].

)DOC");
  }
};

class LoDResetGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) shouldn't be null.");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<framework::LoDTensor>("X")->type()),
        ctx.device_context());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(lod_reset, ops::LoDResetOp, ops::LoDResetOpMaker, lod_reset_grad,
            ops::LoDResetGradOp);
REGISTER_OP_CPU_KERNEL(lod_reset,
                       ops::LoDResetKernel<paddle::platform::CPUPlace, float>,
                       ops::LoDResetKernel<paddle::platform::CPUPlace, double>);
REGISTER_OP_CPU_KERNEL(
    lod_reset_grad, ops::LoDResetGradKernel<paddle::platform::CPUPlace, float>,
    ops::LoDResetGradKernel<paddle::platform::CPUPlace, double>);
