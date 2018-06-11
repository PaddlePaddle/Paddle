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

#include "paddle/fluid/operators/cross_entropy_op.h"

namespace paddle {
namespace operators {

class CrossEntropyOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("Y"), "Output(Y) should be not null.");

    auto x_dims = ctx->GetInputDim("X");
    auto label_dims = ctx->GetInputDim("Label");
    PADDLE_ENFORCE_EQ(x_dims.size(), 2UL, "Input(X)'s rank should be 2.");
    PADDLE_ENFORCE_EQ(label_dims.size(), 2UL,
                      "Input(Label)'s rank should be 2.");
    PADDLE_ENFORCE_EQ(x_dims[0], label_dims[0],
                      "The 1st dimension of Input(X) and Input(Label) should "
                      "be equal.");
    if (ctx->Attrs().Get<bool>("soft_label")) {
      PADDLE_ENFORCE_EQ(x_dims[1], label_dims[1],
                        "If Attr(soft_label) == true, the 2nd dimension of "
                        "Input(X) and Input(Label) should be equal.");
    } else {
      PADDLE_ENFORCE_EQ(label_dims[1], 1UL,
                        "If Attr(softLabel) == false, the 2nd dimension of "
                        "Input(Label) should be 1.");
    }

    ctx->SetOutputDim("Y", {x_dims[0], 1});
    ctx->ShareLoD("X", /*->*/ "Y");
  }

 protected:
  // Explicitly set that the data type of computation kernel of cross_entropy
  // is determined by its input "X".
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("X")->type()),
        ctx.device_context());
  }
};

class CrossEntropyGradientOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Y")),
                   "Input(Y@GRAD) shoudl be not null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Output(X@GRAD) should be not null.");

    auto x_dims = ctx->GetInputDim("X");
    auto label_dims = ctx->GetInputDim("Label");
    auto dy_dims = ctx->GetInputDim(framework::GradVarName("Y"));
    PADDLE_ENFORCE_EQ(x_dims.size(), 2, "Input(X)'s rank should be 2.");
    PADDLE_ENFORCE_EQ(dy_dims.size(), 2, "Input(Y@Grad)'s rank should be 2.");
    PADDLE_ENFORCE_EQ(label_dims.size(), 2, "Input(Label)'s rank should be 2.");
    PADDLE_ENFORCE_EQ(x_dims[0], label_dims[0],
                      "The 1st dimension of Input(X) and Input(Label) should "
                      "be equal.");
    PADDLE_ENFORCE_EQ(x_dims[0], dy_dims[0],
                      "The 1st dimension of Input(X) and Input(Y@Grad) should "
                      "be equal.");
    PADDLE_ENFORCE_EQ(dy_dims[1], 1,
                      "The 2nd dimension of Input(Y@Grad) should be 1.");
    if (ctx->Attrs().Get<bool>("soft_label")) {
      PADDLE_ENFORCE_EQ(x_dims[1], label_dims[1],
                        "When Attr(soft_label) == true, the 2nd dimension of "
                        "Input(X) and Input(Label) should be equal.");
    } else {
      PADDLE_ENFORCE_EQ(label_dims[1], 1,
                        "When Attr(soft_label) == false, the 2nd dimension of "
                        "Input(Label) should be 1.");
    }
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    ctx->ShareLoD("X", framework::GradVarName("X"));
  }

 protected:
  // Explicitly set that the data type of computation kernel of cross_entropy
  // is determined by its input "X".
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("X")->type()),
        ctx.device_context());
  }
};

class CrossEntropyOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, default Tensor<float>), a 2-D tensor with shape [N x D],"
             " where N is the batch size and D is the number of classes. "
             "This input is a probability computed by the previous operator, "
             "which is almost always the result of a softmax operator.");
    AddInput("Label",
             "(Tensor), the ground truth which is a 2-D tensor. When "
             "soft_label is set to false, Label is a Tensor<int64> with shape "
             "[N x 1]. When soft_label is set to true, Label is a "
             "Tensor<float/double> with shape [N x D].");
    AddOutput("Y",
              "(Tensor, default Tensor<float>), a 2-D tensor with shape "
              "[N x 1]. The cross entropy loss.")
        .Reuse("X");
    AddAttr<bool>("soft_label",
                  "(bool, default false), a flag indicating whether to "
                  "interpretate the given labels as soft labels.")
        .SetDefault(false);
    AddComment(R"DOC(
CrossEntropy Operator.

It supports both standard cross-entropy and soft-label cross-entropy loss
computation.
1) One-hot cross-entropy:
    soft_label = false, Label[i, 0] indicates the class index for sample i:

                $Y[i] = -\log(X[i, Label[i]])$

2) Soft-label cross-entropy:
    soft_label = true, Label[i, j] indicates the soft label of class j
    for sample i:

                $Y[i] = \sum_j{-Label[i, j] * log(X[i, j])}$

   Please make sure that in this case the summuation of each row of Label
   equals one.

3) One-hot cross-entropy with vecterized Input(Label):
     As a special case of 2), when each row of Input(Label) has only one
     non-zero element (equals 1), soft-label cross-entropy degenerates to a
     one-hot cross-entropy with one-hot label representation.

Both the input X and Label can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD information with input X.

)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CPUCtx = paddle::platform::CPUDeviceContext;

REGISTER_OPERATOR(cross_entropy, ops::CrossEntropyOp, ops::CrossEntropyOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);
REGISTER_OPERATOR(cross_entropy_grad, ops::CrossEntropyGradientOp);
REGISTER_OP_CPU_KERNEL(cross_entropy, ops::CrossEntropyOpKernel<CPUCtx, float>,
                       ops::CrossEntropyOpKernel<CPUCtx, double>);
REGISTER_OP_CPU_KERNEL(cross_entropy_grad,
                       ops::CrossEntropyGradientOpKernel<CPUCtx, float>,
                       ops::CrossEntropyGradientOpKernel<CPUCtx, double>);
