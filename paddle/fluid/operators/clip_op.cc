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

#include "paddle/fluid/operators/clip_op.h"
#include <memory>

namespace paddle {
namespace operators {

class ClipOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of ClipOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ClipOp should not be null.");
    auto x_dims = ctx->GetInputDim("X");
    auto max = ctx->Attrs().Get<float>("max");
    auto min = ctx->Attrs().Get<float>("min");
    PADDLE_ENFORCE_LT(min, max, "max should be greater than min.");
    ctx->SetOutputDim("Out", x_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

template <typename AttrType>
class ClipOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor)The input of clip op."
             "The number of dimensions must be between [1, 9].");
    AddOutput("Out", "(Tensor)The output of clip op with shape as input(X)");
    AddAttr<AttrType>(
        "min", "(float)Minimum value, under which element is replaced by min.");
    AddAttr<AttrType>(
        "max", "(float)Maximum value, above which element is replaced by max");
    AddComment(R"DOC(
Clip Operator.

The clip operator limits the value of given input within an interval. The
interval is specified with arguments 'min' and 'max':

$$
Out = \min(\max(X, min), max)
$$

)DOC");
  }
};

class ClipOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
    auto x_dims = ctx->GetInputDim("X");
    if (ctx->HasOutput(framework::GradVarName("X"))) {
      ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    }
  }
};

class ClipGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("clip_grad");
    op->SetInput("X", Input("X"));
    op->SetInput(framework::GradVarName("Out"), OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));
    op->SetAttrMap(Attrs());
    return op;
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(clip, ops::ClipOp, ops::ClipOpMaker<float>,
                  ops::ClipGradOpDescMaker);
REGISTER_OPERATOR(clip_grad, ops::ClipOpGrad);
REGISTER_OP_CPU_KERNEL(
    clip, ops::ClipKernel<paddle::platform::CPUDeviceContext, float>);
REGISTER_OP_CPU_KERNEL(
    clip_grad, ops::ClipGradKernel<paddle::platform::CPUDeviceContext, float>);
