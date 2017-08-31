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

#include "paddle/operators/cos_sim_op.h"

namespace paddle {
namespace operators {

using framework::Tensor;

class CosSimOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"), "Input(X) should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Y"), "Input(Y) should not be null.");
    PADDLE_ENFORCE_EQ(ctx.Input<Tensor>("X")->dims(),
                      ctx.Input<Tensor>("Y")->dims(),
                      "Dimensions of Input(X) and Input(Y) must be the same.");

    auto dims = ctx.Input<Tensor>("X")->dims();
    ctx.Output<Tensor>("Out")->Resize({dims[0], 1});
  }
};

class CosSimOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  CosSimOpMaker(framework::OpProto *proto, framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The first input of cos_sim op.");
    AddInput("Y", "The second input of cos_sim op.");
    AddOutput("Out", "The output of cos_sim op.");
    AddComment(R"DOC(
Cosine Similarity Operator.

The equation is: Out = X^T * Y / (sqrt(X^T * X) * sqrt(Y^T * Y))
)DOC");
  }
};

class CosSimOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"), "Input(X) should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Y"), "Input(Y) should not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("Out")),
                            "Input(Out@GRAD) should not be null.");

    auto x_dims = ctx.Input<Tensor>("X")->dims();
    auto y_dims = ctx.Input<Tensor>("Y")->dims();
    auto out_dims = ctx.Input<Tensor>(framework::GradVarName("Out"))->dims();
    PADDLE_ENFORCE_EQ(x_dims, y_dims,
                      "Dimensions of Input(X) and Input(Y) must be the same.");
    PADDLE_ENFORCE_EQ(out_dims[0], x_dims[0],
                      "1st dimension of Out@GRAD must equal to Input(X)");
    PADDLE_ENFORCE_EQ(out_dims[1], 1,
                      "1st dimension of Out@GRAD must equal to Input(X)");

    auto *x_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto *y_grad = ctx.Output<Tensor>(framework::GradVarName("Y"));
    x_grad->Resize(x_dims);
    y_grad->Resize(y_dims);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(cos_sim, ops::CosSimOp, ops::CosSimOpMaker, cos_sim_grad,
            ops::CosSimOpGrad);
REGISTER_OP_CPU_KERNEL(cos_sim,
                       ops::CosSimKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(
    cos_sim_grad, ops::CosSimGradKernel<paddle::platform::CPUPlace, float>);
