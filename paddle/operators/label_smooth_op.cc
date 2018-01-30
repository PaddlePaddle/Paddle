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

#include "paddle/operators/label_smooth_op.h"

namespace paddle {
namespace operators {

class LabelSmoothOp : public framework::OperatorWithKernel {
 public:
  LabelSmoothOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of LabelSmoothOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of LabelSmoothOp should not be null.");
    auto in_dims = ctx->GetInputDim("X");
    ctx->ShareLoD("X", /*->*/ "Out");
    ctx->SetOutputDim("Out", in_dims);
  }
};

class LabelSmoothOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  LabelSmoothOpMaker(OpProto *proto, OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The input label of LabelSmooth operator.");
    AddOutput("Out", "The smoothed label of LabelSmooth operator.");
    AddAttr<float>("epsilon",
                   "(float, default 0.0f)"
                   "The smoothing parameter of LabelSmooth operator.")
        .SetDefault(0.0f);
    AddComment(R"DOC(
LabelSmooth Operator.

)DOC");
  }
};

class LabelSmoothGradOp : public framework::OperatorWithKernel {
 public:
  LabelSmoothGradOp(const std::string &type,
                    const framework::VariableNameMap &inputs,
                    const framework::VariableNameMap &outputs,
                    const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) shouldn't be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) shouldn't be null.");
    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};

}  // namespace operators
}  // namespace paddle
namespace ops = paddle::operators;

REGISTER_OP(label_smooth, ops::LabelSmoothOp, ops::LabelSmoothOpMaker,
            label_smooth_grad, ops::LabelSmoothGradOp);
REGISTER_OP_CPU_KERNEL(
    label_smooth,
    ops::LabelSmoothKernel<paddle::platform::CPUDeviceContext, float>,
    ops::LabelSmoothKernel<paddle::platform::CPUDeviceContext, double>);
REGISTER_OP_CPU_KERNEL(
    label_smooth_grad,
    ops::LabelSmoothGradKernel<paddle::platform::CPUDeviceContext, float>,
    ops::LabelSmoothGradKernel<paddle::platform::CPUDeviceContext, double>);
