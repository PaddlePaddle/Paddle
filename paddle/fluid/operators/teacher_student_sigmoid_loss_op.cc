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

#include "paddle/fluid/operators/teacher_student_sigmoid_loss_op.h"

#include <memory>

#include "paddle/fluid/operators/math/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

class TeacherStudentSigmoidLossOp : public framework::OperatorWithKernel {
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
    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(x_dims[0], label_dims[0],
                        "The 1st dimension of Input(X) and Input(Label) should "
                        "be equal.");
      PADDLE_ENFORCE_EQ(label_dims[1], 1UL,
                        "The 2nd dimension of "
                        "Input(Label) should be 1.");
    }
    ctx->SetOutputDim("Y", {x_dims[0], 1});
    ctx->ShareLoD("X", /*->*/ "Y");
  }

 protected:
  // Explicitly set that the data type of computation kernel of
  // teacher_student_sigmoid_loss
  // is determined by its input "X".
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   ctx.device_context());
  }
};

class TeacherStudentSigmoidLossGradOpDescMaker
    : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());

    op->SetType("teacher_student_sigmoid_loss_grad");

    op->SetInput("X", Input("X"));
    op->SetInput("Label", Input("Label"));
    op->SetInput(framework::GradVarName("Y"), OutputGrad("Y"));

    op->SetOutput(framework::GradVarName("X"), InputGrad("X"));

    op->SetAttrMap(Attrs());
    return op;
  }
};

class TeacherStudentSigmoidLossGradientOp
    : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Y")),
                   "Input(Y@GRAD) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("X")),
                   "Output(X@GRAD) should be not null.");

    auto x_dims = ctx->GetInputDim("X");
    auto label_dims = ctx->GetInputDim("Label");
    auto dy_dims = ctx->GetInputDim(framework::GradVarName("Y"));
    PADDLE_ENFORCE_EQ(x_dims.size(), 2, "Input(X)'s rank should be 2.");
    PADDLE_ENFORCE_EQ(dy_dims.size(), 2, "Input(Y@Grad)'s rank should be 2.");
    PADDLE_ENFORCE_EQ(label_dims.size(), 2, "Input(Label)'s rank should be 2.");
    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(x_dims[0], label_dims[0],
                        "The 1st dimension of Input(X) and Input(Label) should "
                        "be equal.");
      PADDLE_ENFORCE_EQ(
          x_dims[0], dy_dims[0],
          "The 1st dimension of Input(X) and Input(Y@Grad) should "
          "be equal.");
      PADDLE_ENFORCE_EQ(dy_dims[1], 1,
                        "The 2nd dimension of Input(Y@Grad) should be 1.");
      PADDLE_ENFORCE_EQ(label_dims[1], 1,
                        "When Attr(soft_label) == false, the 2nd dimension of "
                        "Input(Label) should be 1.");
    }
    ctx->SetOutputDim(framework::GradVarName("X"), x_dims);
    ctx->ShareLoD("X", framework::GradVarName("X"));
  }

 protected:
  // Explicitly set that the data type of computation kernel of
  // teacher_student_sigmoid_loss
  // is determined by its input "X".
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(ctx.Input<Tensor>("X")->type(),
                                   ctx.device_context());
  }
};

class TeacherStudentSigmoidLossOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor, default Tensor<float>), a 2-D tensor with shape [N x 1],"
             " where N is the batch size and D is the output. "
             "This input is a probability computed by the previous operator, "
             "which is almost always the result of a softmax operator.");
    AddInput("Label",
             "(Tensor), the ground truth which is a 2-D tensor. "
             "Label is a Tensor<float> with shape [N x 1]. ");
    AddOutput("Y",
              "(Tensor, default Tensor<float>), a 2-D tensor with shape "
              "[N x 1]. The teacher student sigmoid loss.");
    AddAttr<float>(
        "soft_max_up_bound",
        "fp32, if input > soft_max_up_bound, input will be bound, default 15.0")
        .SetDefault(15.0);
    AddAttr<float>("soft_max_lower_bound",
                   "fp32, if input < soft_max_lower_bound, input will be "
                   "bound, default -15.0")
        .SetDefault(-15.0);
    AddComment(R"DOC(
TeacherStudentSigmoidLoss Operator.

It's similarity to SigmoidCrossEntropyWithLogits Operator. The difference is that
we add another label(z') to original.
        loss = max(x, 0) - x * z + log(1 + exp(-abs(x))) + max(x, 0) - x * z' + log(1 + exp(-abs(x)))
        z is click or not
        z' is teacher value 
        label = {-2, -1, [0, 2]}
        when z' is not exist, clk = 0 : label = -2;
        when z' is not exist, clk = 1 : label = -1;
        when z' is exist , clk = 0 : label = 0 + z';
        when z' is exist    , clk = 1 : label = 1 + z';

)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(teacher_student_sigmoid_loss,
                  ops::TeacherStudentSigmoidLossOp,
                  ops::TeacherStudentSigmoidLossOpMaker,
                  ops::TeacherStudentSigmoidLossGradOpDescMaker);

REGISTER_OPERATOR(teacher_student_sigmoid_loss_grad,
                  ops::TeacherStudentSigmoidLossGradientOp);

REGISTER_OP_CPU_KERNEL(teacher_student_sigmoid_loss,
                       ops::TeacherStudentSigmoidLossOpKernel<float>,
                       ops::TeacherStudentSigmoidLossOpKernel<double>);

REGISTER_OP_CPU_KERNEL(teacher_student_sigmoid_loss_grad,
                       ops::TeacherStudentSigmoidLossGradOpKernel<float>,
                       ops::TeacherStudentSigmoidLossGradOpKernel<double>);
