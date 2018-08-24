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

#include "paddle/fluid/operators/pad_constant_batch_size_like_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class PadConstantBatchSizeLikeOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(
        ctx->HasInput("X"),
        "Input(X) of PadConstantBatchSizeLikeOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasInput("Y"),
        "Input(Y) of PadConstantBatchSizeLikeOp should not be null.");
    PADDLE_ENFORCE(
        ctx->HasOutput("Out"),
        "Output(Out) of PadConstantBatchSizeLikeOp should not be null.");

    auto x_dim = ctx->GetInputDim("X");
    auto y_dim = ctx->GetInputDim("Y");

    PADDLE_ENFORCE_EQ(x_dim.size(), y_dim.size(),
                      "The dimention of X and Y should be the same.");

    for (int i = 0; i < x_dim.size(); ++i) {
      if (i == 0) {
        PADDLE_ENFORCE_GE(x_dim[i], y_dim[i]);
      } else {
        PADDLE_ENFORCE_EQ(x_dim[i], y_dim[i], "");
      }
    }
    ctx->SetOutputDim("Out", x_dim);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class PadConstantBatchSizeLikeOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input of pad op. "
             "The input should be a k-D tensor(k > 0 and k < 7)");
    AddInput("Y",
             "The input of pad op. "
             "The input should be a k-D tensor(k > 0 and k < 7)");
    AddOutput("Out",
              "The output of pad op. "
              "A tensor with the same shape as X.");
    AddAttr<float>("pad_value",
                   "(float, default 0.0) "
                   "The value to fill the padded areas.")
        .SetDefault(0.0f);
    AddComment(R"DOC(
PadConstantBatchSizeLikeOp Operator.

PadConstantBatchSizeLikeOp input into output, as specified by paddings and pad_value.
The input should be a k-D tensor(k > 0 and k < 7). As an example:

Given:

X = [[1, 2],
     [3, 4],
     [1, 2],
     [3, 4]]],

Y = [[5, 6],
     [7, 8]],

and

pad_value = 0,

we have:

Out = [[5, 6],
       [7, 8],
       [0, 0],
       [0, 0]]

)DOC");
  }
};

class PadConstantBatchSizeLikeOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput("Y"), "Input(Y) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
    auto y_dims = ctx->GetInputDim("Y");
    auto y_grad_name = framework::GradVarName("Y");
    if (ctx->HasOutput(y_grad_name)) {
      ctx->SetOutputDim(y_grad_name, y_dims);
      ctx->ShareLoD(y_grad_name, /*->*/ "Y");
    }
  }
};

class PadConstantBatchSizeLikeOpGradMaker
    : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *bind = new framework::OpDesc();
    bind->SetType("pad_constant_batch_size_like_grad");
    bind->SetInput("X", Input("X"));
    bind->SetInput("Y", Input("Y"));
    bind->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    bind->SetOutput(framework::GradVarName("Y"), InputGrad("Y"));
    bind->SetAttrMap(Attrs());
    return std::unique_ptr<framework::OpDesc>(bind);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(pad_constant_batch_size_like, ops::PadConstantBatchSizeLikeOp,
                  ops::PadConstantBatchSizeLikeOpMaker,
                  ops::PadConstantBatchSizeLikeOpGradMaker);
REGISTER_OPERATOR(pad_constant_batch_size_like_grad,
                  ops::PadConstantBatchSizeLikeOpGrad);
REGISTER_OP_CPU_KERNEL(pad_constant_batch_size_like,
                       ops::PadConstantBatchSizeLikeKernel<
                           paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(pad_constant_batch_size_like_grad,
                       ops::PadConstantBatchSizeLikeGradKernel<
                           paddle::platform::CPUDeviceContext, float>);
