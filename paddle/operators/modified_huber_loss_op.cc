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

#include "paddle/operators/modified_huber_loss_op.h"

namespace paddle {
namespace operators {

class ModifiedHuberLossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext& context) const override {
    PADDLE_ENFORCE_NOT_NULL(context.InputVar("X"), "X must be initialized.");
    PADDLE_ENFORCE_NOT_NULL(context.InputVar("Y"), "Y must be initialized.");

    auto* x = context.Input<Tensor>("X");
    auto* y = context.Input<Tensor>("Y");

    PADDLE_ENFORCE_EQ(x->dims(), y->dims(),
                      "Dimensions of X and Y must be the same.");
    PADDLE_ENFORCE_EQ(framework::arity(x->dims()), 2,
                      "Tensor rank of X must be 2.");
    PADDLE_ENFORCE_EQ(x->dims()[1], 1, "Second dimension of X must be 1.");

    context.Output<Tensor>("intermediate_val")->Resize(x->dims());
    context.Output<Tensor>("Out")->Resize({x->dims()[0], 1});
  }
};

class ModifiedHuberLossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  ModifiedHuberLossOpMaker(framework::OpProto* proto,
                           framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "");
    AddInput("Y", "");
    AddOutput("intermediate_val", "").AsIntermediate();
    AddOutput("Out", "");
    AddComment("");
  }
};

class ModifiedHuberLossGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* y = context.Input<Tensor>("Y");
    auto* intermediate_val = context.Input<Tensor>("intermediate_val");
    auto* out_grad = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* x_grad = context.Output<Tensor>(framework::GradVarName("X"));
    auto* y_grad = context.Output<Tensor>(framework::GradVarName("Y"));

    PADDLE_ENFORCE_NOT_NULL(x, "Input X must not be null.");
    PADDLE_ENFORCE_NOT_NULL(y, "Target Y must not be null.");
    PADDLE_ENFORCE_NOT_NULL(intermediate_val,
                            "Intermediate value must not be null.");
    PADDLE_ENFORCE_NOT_NULL(out_grad, "Out gradient must not be null.");

    PADDLE_ENFORCE_EQ(
        intermediate_val->dims(), x->dims(),
        "Dimension of X and intermediate value must be the same.");
    PADDLE_ENFORCE_EQ(
        out_grad->dims(), x->dims(),
        "Dimension of Out gradient and X must be the same (N*1).");

    if (x_grad) x_grad->Resize(x->dims());
    if (y_grad) y_grad->Resize(y->dims());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(modified_huber_loss, ops::ModifiedHuberLossOp,
            ops::ModifiedHuberLossOpMaker, modified_huber_loss_grad,
            ops::ModifiedHuberLossGradOp);

REGISTER_OP_CPU_KERNEL(
    modified_huber_loss,
    ops::ModifiedHuberLossKernel<paddle::platform::CPUPlace, float>);
REGISTER_OP_CPU_KERNEL(modified_huber_loss_grad,
                       ops::ModifiedHuberLossGradCPUKernel<float>);
