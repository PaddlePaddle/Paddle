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

#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

class TeacherStudentSigmoidLossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("X"), "Input", "X", "teacher_student_sigmoid_loss");
    OP_INOUT_CHECK(ctx->HasInput("Label"),
                   "Input",
                   "Label",
                   "teacher_student_sigmoid_loss");
    OP_INOUT_CHECK(
        ctx->HasOutput("Y"), "Output", "Y", "teacher_student_sigmoid_loss");

    auto x_dims = ctx->GetInputDim("X");
    auto label_dims = ctx->GetInputDim("Label");
    PADDLE_ENFORCE_EQ(x_dims.size(),
                      2UL,
                      platform::errors::InvalidArgument(
                          "Input(X)'s rank should be 2. But received: "
                          "Input(X)'s rank is [%d]",
                          x_dims.size()));
    PADDLE_ENFORCE_EQ(label_dims.size(),
                      2UL,
                      platform::errors::InvalidArgument(
                          "Input(Label)'s rank should be 2. But "
                          "received Input(Label)'s rank is [%d]",
                          label_dims.size()));
    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(
          x_dims[0],
          label_dims[0],
          platform::errors::InvalidArgument(
              "The 1st dimension of Input(X) and Input(Label) should "
              "be equal. The difference is [%d]: [%d]",
              x_dims[0],
              label_dims[0]));
      PADDLE_ENFORCE_EQ(label_dims[1],
                        1UL,
                        platform::errors::InvalidArgument(
                            "The 2nd dimension of "
                            "Input(Label) should be 1. But received "
                            "Input(Label)'s 2nd dim is [%d]",
                            label_dims[1]));
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
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

template <typename T>
class TeacherStudentSigmoidLossGradOpMaker
    : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("teacher_student_sigmoid_loss_grad");

    op->SetInput("X", this->Input("X"));
    op->SetInput("Label", this->Input("Label"));
    op->SetInput(framework::GradVarName("Y"), this->OutputGrad("Y"));

    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));

    op->SetAttrMap(this->Attrs());
  }
};

class TeacherStudentSigmoidLossGradientOp
    : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("X"), "Input", "X", "teacher_student_sigmoid_loss_grad");
    OP_INOUT_CHECK(ctx->HasInput("Label"),
                   "Input",
                   "X",
                   "teacher_student_sigmoid_loss_grad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Y")),
                   "Input",
                   "Y@Grad",
                   "teacher_student_sigmoid_loss_grad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")),
                   "Input",
                   "X@Grad",
                   "teacher_student_sigmoid_loss_grad");

    auto x_dims = ctx->GetInputDim("X");
    auto label_dims = ctx->GetInputDim("Label");
    auto dy_dims = ctx->GetInputDim(framework::GradVarName("Y"));
    PADDLE_ENFORCE_EQ(
        x_dims.size(),
        2,
        platform::errors::InvalidArgument(
            "Input(X)'s rank should be 2. But received Input(X)'s rank is [%d]",
            x_dims.size()));
    PADDLE_ENFORCE_EQ(dy_dims.size(),
                      2,
                      platform::errors::InvalidArgument(
                          "Input(Y@Grad)'s rank should be 2. But received "
                          "Input(Y@Grad)'s rank is [%d]",
                          dy_dims.size()));
    PADDLE_ENFORCE_EQ(label_dims.size(),
                      2,
                      platform::errors::InvalidArgument(
                          "Input(Label)'s rank should be 2. But received "
                          "Input(Y@Grad)'s rank is [%d]",
                          label_dims.size()));
    if (ctx->IsRuntime()) {
      PADDLE_ENFORCE_EQ(
          x_dims[0],
          label_dims[0],
          platform::errors::InvalidArgument(
              "The 1st dimension of Input(X) and Input(Label) should "
              "be equal. The difference is [%d]: [%d]",
              x_dims[0],
              label_dims[0]));
      PADDLE_ENFORCE_EQ(
          x_dims[0],
          dy_dims[0],
          platform::errors::InvalidArgument(
              "The 1st dimension of Input(X) and Input(Y@Grad) should "
              "be equal. The difference is [%d]: [%d]",
              x_dims[0],
              dy_dims[0]));
      PADDLE_ENFORCE_EQ(
          dy_dims[1],
          1,
          platform::errors::InvalidArgument(
              "The 2nd dimension of Input(Y@Grad) should be 1. "
              "But received Input(Y@Grad)'s 2nd dimension is [%d]",
              dy_dims[1]));
      PADDLE_ENFORCE_EQ(
          label_dims[1],
          1,
          platform::errors::InvalidArgument(
              "When Attr(soft_label) == false, the 2nd dimension of "
              "Input(Label) should be 1. But received Input(Label)'s 2nd "
              "dimemsion "
              "is [%d]",
              label_dims[1]));
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
    return framework::OpKernelType(
        OperatorWithKernel::IndicateVarDataType(ctx, "X"),
        ctx.device_context());
  }
};

class TeacherStudentSigmoidLossOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(phi::DenseTensor, default phi::DenseTensor<float>), a 2-D "
             "tensor with shape [N x 1],"
             " where N is the batch size and D is the output. "
             "This input is a probability computed by the previous operator, "
             "which is almost always the result of a softmax operator.");
    AddInput("Label",
             "(phi::DenseTensor), the ground truth which is a 2-D tensor. "
             "Label is a phi::DenseTensor<float> with shape [N x 1]. ");
    AddOutput("Y",
              "(phi::DenseTensor, default phi::DenseTensor<float>), a 2-D "
              "tensor with shape "
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
REGISTER_OPERATOR(
    teacher_student_sigmoid_loss,
    ops::TeacherStudentSigmoidLossOp,
    ops::TeacherStudentSigmoidLossOpMaker,
    ops::TeacherStudentSigmoidLossGradOpMaker<paddle::framework::OpDesc>,
    ops::TeacherStudentSigmoidLossGradOpMaker<paddle::imperative::OpBase>);

REGISTER_OPERATOR(teacher_student_sigmoid_loss_grad,
                  ops::TeacherStudentSigmoidLossGradientOp);

REGISTER_OP_CPU_KERNEL(teacher_student_sigmoid_loss,
                       ops::TeacherStudentSigmoidLossOpKernel<float>,
                       ops::TeacherStudentSigmoidLossOpKernel<double>);

REGISTER_OP_CPU_KERNEL(teacher_student_sigmoid_loss_grad,
                       ops::TeacherStudentSigmoidLossGradOpKernel<float>,
                       ops::TeacherStudentSigmoidLossGradOpKernel<double>);
