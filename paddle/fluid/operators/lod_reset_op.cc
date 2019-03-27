/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/lod_reset_op.h"
#include <memory>

namespace paddle {
namespace operators {

class LoDResetOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of LoDResetOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of LoDResetOp should not be null.");

    if (!ctx->HasInput("Y")) {
      auto level0 = ctx->Attrs().Get<std::vector<int>>("target_lod");
      PADDLE_ENFORCE_GT(level0.size(), 1,
                        "If Input(Y) not provided, the target lod should be "
                        "specified by attribute `target_lod`.");
    } else {
      ctx->ShareLoD("Y", "Out");
    }

    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(ctx.Input<framework::LoDTensor>("X")->type(),
                                   ctx.device_context());
  }
};

class LoDResetOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, LoDTensor) Input variable of LoDResetOp which "
             "could be a Tensor or LoDTensor, where the data of output "
             "variable inherits from.");
    AddInput("Y",
             "(Tensor, LoDTensor, optional) If provided and Y is LoDTensor, "
             "lod of Input(Y) would be considered as the target lod first, "
             "otherwise data of Input(Y) would be considered as the "
             "target lod.")
        .AsDispensable();
    AddOutput("Out",
              "(LoDTensor) Output variable of LoDResetOp which should be a "
              "LoDTensor.");
    AddAttr<std::vector<int>>("target_lod",
                              "The target level 0 LoD from Attr().")
        .SetDefault(std::vector<int>{});
    AddComment(R"DOC(LoDReset operator

Set LoD of `X` to a new one specified by `Y` or attribute `target_lod`. When `Y`
provided and `Y` is a LoDTensor, `Y.lod` would be considered as target LoD
first, otherwise `Y.data` would be considered as target LoD. If `Y` is not
provided, target LoD should be specified by attribute `target_lod`.
If target LoD is specified by `Y.data` or `target_lod`, only one level LoD
is supported.

Example 1:

Given a 1-level LoDTensor input(X):
    X.lod =  [[ 0,     2,                   5      6 ]]
    X.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    X.dims = [6, 1]

attr(target_lod): [0, 4, 6]

then we get a 1-level LoDTensor:
    Out.lod =  [[ 0,                   4,            6 ]]
    Out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    Out.dims = [6, 1]

Example 2:

Given a 1-level LoDTensor input(X):
    X.lod =  [[ 0,     2,                   5      6 ]]
    X.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    X.dims = [6, 1]

input(Y) is a Tensor:
    Y.data = [[0, 2, 6]]
    Y.dims = [1, 3]

then we get a 1-level LoDTensor:
    Out.lod =  [[ 0,     2,                          6 ]]
    Out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    Out.dims = [6, 1]

Example 3:

Given a 1-level LoDTensor input(X):
    X.lod =  [[ 0,      2,                   5     6 ]]
    X.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    X.dims = [6, 1]

input(Y) is a 2-level LoDTensor:
    Y.lod =  [[0, 2, 4], [0, 2, 5, 6]]
    Y.data = [[1.1], [2.1], [3.1], [4.1], [5.1], [6.1]]
    Y.dims = [6, 1]

then we get a 2-level LoDTensor:
    Out.lod =  [[0, 2, 4], [0, 2, 5, 6]]
    Out.data = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]
    Out.dims = [6, 1]

)DOC");
  }
};

class LoDResetGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of LoDResetGradOp should not be null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@Grad) of LoDResetGradOp should not be null.");

    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, ctx->GetInputDim("X"));
      ctx->ShareLoD("X", /*->*/ x_grad_name);
    }
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    return framework::OpKernelType(
        ctx.Input<framework::LoDTensor>(framework::GradVarName("Out"))->type(),
        ctx.device_context());
  }
};

class LoDResetGradDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("lod_reset_grad");
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetInput("X", Input("X"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetAttrMap(Attrs());
    return op;
  }
};

DECLARE_NO_NEED_BUFFER_VARS_INFERENCE(LoDResetGradNoNeedBufferVarInference,
                                      "X");

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(lod_reset, ops::LoDResetOp, ops::LoDResetOpMaker,
                  ops::LoDResetGradDescMaker);
REGISTER_OPERATOR(lod_reset_grad, ops::LoDResetGradOp,
                  ops::LoDResetGradNoNeedBufferVarInference);
REGISTER_OP_CPU_KERNEL(
    lod_reset, ops::LoDResetKernel<paddle::platform::CPUPlace, float>,
    ops::LoDResetKernel<paddle::platform::CPUPlace, double>,
    ops::LoDResetKernel<paddle::platform::CPUPlace, int>,
    ops::LoDResetKernel<paddle::platform::CPUPlace, int64_t>);
REGISTER_OP_CPU_KERNEL(
    lod_reset_grad, ops::LoDResetGradKernel<paddle::platform::CPUPlace, float>,
    ops::LoDResetGradKernel<paddle::platform::CPUPlace, double>,
    ops::LoDResetGradKernel<paddle::platform::CPUPlace, int>,
    ops::LoDResetGradKernel<paddle::platform::CPUPlace, int64_t>);
