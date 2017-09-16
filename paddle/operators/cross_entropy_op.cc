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

#include "paddle/operators/cross_entropy_op.h"

namespace paddle {
namespace operators {

using framework::LoDTensor;

class CrossEntropyOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"),
                            "Input(X) of CrossEntropyOp must not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Label"),
                            "Input(Label) of CrossEntropyOp must not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.OutputVar("Y"),
                            "Output(Y) of CrossEntropyOp must not be null.");

    auto *x = ctx.Input<Tensor>("X");
    auto *label = ctx.Input<Tensor>("Label");

    PADDLE_ENFORCE_EQ(x->dims().size(), 2, "X's rank must be 2.");
    PADDLE_ASSERT(label->dims().size() == 1 || label->dims().size() == 2);
    if (label->dims().size() == 2) {
      // soft cross entropy
      PADDLE_ENFORCE_EQ(x->dims(), label->dims());
    } else {
      // normal cross entropy
      PADDLE_ENFORCE_EQ(x->dims()[0], label->dims()[0]);
    }
    ctx.Output<LoDTensor>("Y")->Resize({x->dims()[0], 1});
  }
};

class CrossEntropyGradientOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"),
                            "Input(X) of CrossEntropyOp must not be null.");

    auto dx = ctx.Output<LoDTensor>(framework::GradVarName("X"));
    auto x = ctx.Input<Tensor>("X");

    dx->Resize(x->dims());
  }
};

class CrossEntropyOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  CrossEntropyOpMaker(framework::OpProto *proto,
                      framework::OpAttrChecker *op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "The first input of CrossEntropyOp");
    AddInput("Label", "The second input of CrossEntropyOp");
    AddOutput("Y", "The output of CrossEntropyOp");
    AddComment(R"DOC(
CrossEntropy Operator.

The second input (Label tensor) supports two kinds of shapes:
1) Rank(Label) = 1, Label[i] indicates the class index for sample i:

                Y[i] = -log(X[i, Label[i]])

2) Rank(Label) = 2, Label[i, j] indicates the soft label of class j
   for sample i:

                Y[i] = \sum_j{-Label[i, j] * log(X[i, j])}

   Please make sure that in this case the summuation of each row of Label
   equals one. If each row of Label has only one non-zero element (equals 1),
   it degenerates to a standard one-hot representation.
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(cross_entropy, ops::CrossEntropyOp, ops::CrossEntropyOpMaker,
            cross_entropy_grad, ops::CrossEntropyGradientOp);
REGISTER_OP_CPU_KERNEL(cross_entropy, ops::CrossEntropyOpKernel<float>);
REGISTER_OP_CPU_KERNEL(cross_entropy_grad,
                       ops::CrossEntropyGradientOpKernel<float>);
