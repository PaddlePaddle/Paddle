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

class CrossEntropyOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"), "Input(X) must not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Label"),
                            "Input(Label) must not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.OutputVar("Y"), "Output(Y) must not be null.");

    auto x = ctx.Input<Tensor>("X");
    auto label = ctx.Input<Tensor>("Label");
    PADDLE_ENFORCE_EQ(x->dims().size(), 2, "Input(X)'s rank must be 2.");
    PADDLE_ENFORCE_EQ(label->dims().size(), 2,
                      "Input(Label)'s rank must be 2.");
    PADDLE_ENFORCE_EQ(x->dims()[0], label->dims()[0],
                      "The 1st dimension of Input(X) and Input(Label) must "
                      "be equal.");
    if (ctx.Attr<bool>("soft_label")) {
      PADDLE_ENFORCE_EQ(x->dims()[1], label->dims()[1],
                        "If Attr(soft_label) == true, The 2nd dimension of "
                        "Input(X) and Input(Label) must be equal.");
    } else {
      PADDLE_ENFORCE_EQ(label->dims()[1], 1,
                        "If Attr(soft_label) == false, The 2nd dimension of "
                        "Input(Label) must be 1.");
    }

    ctx.Output<Tensor>("Y")->Resize({x->dims()[0], 1});
    ctx.ShareLoD("X", /*->*/ "Y");
  }
};

class CrossEntropyGradientOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(const framework::InferShapeContext &ctx) const override {
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("X"), "Input(X) must not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar("Label"),
                            "Input(Label) must not be null.");
    PADDLE_ENFORCE_NOT_NULL(ctx.InputVar(framework::GradVarName("Y")),
                            "Input(Y@GRAD) must not be null.");

    auto x = ctx.Input<Tensor>("X");
    auto label = ctx.Input<Tensor>("Label");
    auto dy = ctx.Input<Tensor>(framework::GradVarName("Y"));
    PADDLE_ENFORCE_EQ(x->dims().size(), 2, "Input(X)'s rank must be 2.");
    PADDLE_ENFORCE_EQ(dy->dims().size(), 2, "Input(Y@Grad)'s rank must be 2.");
    PADDLE_ENFORCE_EQ(label->dims().size(), 2,
                      "Input(Label)'s rank must be 2.");
    PADDLE_ENFORCE_EQ(x->dims()[0], label->dims()[0],
                      "The 1st dimension of Input(X) and Input(Label) must "
                      "be equal.");
    PADDLE_ENFORCE_EQ(x->dims()[0], dy->dims()[0],
                      "The 1st dimension of Input(X) and Input(Y@Grad) must "
                      "be equal.");
    PADDLE_ENFORCE_EQ(dy->dims()[1], 1,
                      "The 2nd dimension of Input(Y@Grad) must be 1.");
    if (ctx.Attr<bool>("soft_label")) {
      PADDLE_ENFORCE_EQ(x->dims()[1], label->dims()[1],
                        "If Attr(soft_label) == true, The 2nd dimension of "
                        "Input(X) and Input(Label) must be equal.");
    } else {
      PADDLE_ENFORCE_EQ(label->dims()[1], 1,
                        "If Attr(soft_label) == false, The 2nd dimension of "
                        "Input(Label) must be 1.");
    }

    auto dx = ctx.Output<Tensor>(framework::GradVarName("X"));
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
    AddAttr<bool>("soft_label", "Is soft label. Default zero.")
        .SetDefault(false);

    AddComment(R"DOC(
CrossEntropy Operator.

It supports both standard cross-entropy and soft-label cross-entropy loss
computation.
1) One-hot cross-entropy:
    soft_label = False, Label[i, 0] indicates the class index for sample i:

                Y[i] = -log(X[i, Label[i]])

2) Soft-label cross-entropy:
    soft_label = True, Label[i, j] indicates the soft label of class j
    for sample i:

                Y[i] = \sum_j{-Label[i, j] * log(X[i, j])}

   Please make sure that in this case the summuation of each row of Label
   equals one.

3) One-hot cross-entropy with vecterized Input(Label):
     As a special case of 2), when each row of Input(Label) has only one
     non-zero element (equals 1), soft-label cross-entropy degenerates to a
     one-hot cross-entropy with one-hot label representation.

Both the input `X` and `Label` can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD with input `X`.
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
