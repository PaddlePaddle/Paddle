/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/hardmax_op.h"

namespace paddle {
namespace operators {

class HardmaxOP : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of HardmaxOP should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of HardmaxOP should not be null.");
    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE(x_dims.size() == 2UL,
                   "The input of hardmax op must be a matrix.");
    ctx->SetOutputDim("Out", x_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

template <typename AttrType>
class HardmaxOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  HardmaxOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X","(Tensor)The input of hardmax op."
    AddOutput("Out", "(Tensor)The output of hardmax op with shape as input(X)");
    AddAttr<int>("axis",
                 "(int, default 1 "
                 "List of positive integers,"
                 "describes the axis of the inputs.").SetDefault(1);
    AddComment(R"DOC(
Hardmax Operator.

The operator computes the hardmax (1 for the first maximum value,
and 0 for all others) values for each layer in the batch of the given input. 

)DOC");
  }
};

class HardmaxOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Out"), "Input(Out) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should be not null.");
    PADDLE_ENFORCE_EQ(ctx->GetInputDim("Out"),
                      ctx->GetInputDim(framework::GradVarName("Out")),
                      "Input(Out) and its gradients should have a same shape.");

    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(hardmax, ops::HardmaxOP, ops::HardmaxOpMaker, hardmax_grad,
            ops::HardmaxOpGrad);
REGISTER_OP_CPU_KERNEL(
    hardmax, ops::HardmaxKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    hardmax_grad,
    ops::HardmaxKernel<paddle::platform::CPUDeviceContext, float>);