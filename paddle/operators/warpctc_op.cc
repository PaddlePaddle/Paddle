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

#include "paddle/operators/warpctc_op.h"

namespace paddle {
namespace operators {

class WarpCTCOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Logits"),
                   "Input(Logits) of WarpCTCOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"),
                   "Input(Label) of WarpCTCOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("WarpCTCGrad"),
                   "Output(WarpCTCGrad) of WarpCTCOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Loss"),
                   "Output(Loss) of WarpCTCOp should not be null.");

    auto dims = ctx->GetInputDim("Logits");
    int sequence_width = static_cast<int>(framework::product(dims) / dims[0]);
    int blank = ctx->Attrs().Get<int>("blank");
    PADDLE_ENFORCE((blank > 0) && (blank < sequence_width),
                   "The value of Attr(blank) should be in interval [0, %d).",
                   sequence_width);
  }

  framework::DataType IndicateDataType(
      const framework::ExecutionContext& ctx) const override {
    return framework::ToDataType(ctx.Input<Tensor>("Logits")->type());
  }
};

class WarpCTCOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  WarpCTCOpMaker(framework::OpProto* proto,
                 framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("Logits", "(LodTensor)");
    AddInput("Label", "(LodTensor)");
    AddOutput("WarpCTCGrad", "(Tensor)").AsIntermediate();
    AddOutput("Loss", "(Tensor)");
    AddAttr<int>("blank", "").SetDefault(0);
    AddAttr<bool>("normByTimes", "").SetDefault(false);
    AddComment(R"DOC(
An operator integrating the open-source warp-ctc library
<https://github.com/baidu-research/warp-ctc> to compute connectionist
temporal classification cost.
It can be aliased as softmax_with_ctc, since a softmax is computed to
normlize values for each row of the input tensor, following with a ctc loss.
)DOC");
  }
};

class WarpCTCGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("WarpCTCGrad"),
                   "Input(WarpCTCGrad) of WarpCTCGradOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("Logits")),
                   "Output(Logits@GRAD) of WarpCTCGradOp should not be null.");
  }

  framework::DataType IndicateDataType(
      const framework::ExecutionContext& ctx) const override {
    return framework::ToDataType(ctx.Input<Tensor>("Logits")->type());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(warpctc, ops::WarpCTCOp, ops::WarpCTCOpMaker, warpctc_grad,
            ops::WarpCTCGradOp);
REGISTER_OP_CPU_KERNEL(warpctc,
                       ops::WarpCTCKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    warpctc_grad, ops::WarpCTCGradKernel<paddle::platform::CPUPlace, float>);
