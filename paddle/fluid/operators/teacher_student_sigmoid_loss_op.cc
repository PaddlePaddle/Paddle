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
    PADDLE_ENFORCE_EQ(x_dims[0], label_dims[0],
                      "The 1st dimension of Input(X) and Input(Label) should "
                      "be equal.");
    PADDLE_ENFORCE_EQ(label_dims[1], 1UL,
                      "The 2nd dimension of "
                      "Input(Label) should be 1.");
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
    PADDLE_ENFORCE_EQ(x_dims[0], label_dims[0],
                      "The 1st dimension of Input(X) and Input(Label) should "
                      "be equal.");
    PADDLE_ENFORCE_EQ(x_dims[0], dy_dims[0],
                      "The 1st dimension of Input(X) and Input(Y@Grad) should "
                      "be equal.");
    PADDLE_ENFORCE_EQ(dy_dims[1], 1,
                      "The 2nd dimension of Input(Y@Grad) should be 1.");
    PADDLE_ENFORCE_EQ(label_dims[1], 1,
                      "When Attr(soft_label) == false, the 2nd dimension of "
                      "Input(Label) should be 1.");
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
    AddAttr<float>("soft_max_up_bound", "fp32, default 15.0").SetDefault(15.0);
    AddAttr<float>("soft_max_lower_bound", "fp32, default -15.0")
        .SetDefault(-15.0);
    AddComment(R"DOC(
TeacherStudentSigmoidLoss Operator.
TeacherStudentSigmoidLoss Operator.

It's similarity to SigmoidCrossEntropyWithLogits Operator. The difference is that
we add another label(z') to original.
        loss = max(x, 0) - x * z + log(1 + exp(-abs(x))) + max(x, 0) - x * z' + log(1 + exp(-abs(x)))
        z is click or not
        z' is value q of feed_fine
        label = {-2, -1, [0, 2]}
        when z' is not exist, clk = 0 : label = -2;
        when z' is not exist, clk = 1 : label = -1;
        when z' is exist    , clk = 0 : label = 0 + z';
        when z' is exist    , clk = 1 : label = 1 + z';

)DOC");
  }
};

// template <typename DeviceContext, typename T>
template <typename T>
class TeacherStudentSigmoidLossOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(context.GetPlace()),
                   "This kernel only runs on CPU.");

    Tensor* y = context.Output<Tensor>("Y");
    const Tensor* x = context.Input<Tensor>("X");
    const Tensor* labels = context.Input<Tensor>("Label");
    T* y_data = y->mutable_data<T>(context.GetPlace());
    const T* x_data = x->data<T>();
    const T* label_data = labels->data<T>();
    int64_t batch_size = x->dims()[0];
    // loss = max(x, 0) - x * z + log(1 + exp(-abs(x))) + max(x, 0) - x * z' +
    // log(1 + exp(-abs(x)))
    // z is click or not
    // z' is value q of feed_fine
    // label = {-2, -1, [0, 2]}
    // when z' is not exist, clk = 0 : label = -2;
    // when z' is not exist, clk = 1 : label = -1;
    // when z' is exist    , clk = 0 : label = 0 + z';
    // when z' is exist    , clk = 1 : label = 1 + z';
    for (int i = 0; i < batch_size; ++i) {
      if (label_data[i] < -1.0) {
        y_data[i] = (x_data[i] > 0 ? x_data[i] : 0.0) +
                    log(1.0 + exp(-fabs(x_data[i])));
      } else if (label_data[i] < 0.0) {
        y_data[i] = (x_data[i] > 0 ? x_data[i] : 0.0) - x_data[i] +
                    log(1.0 + exp(-fabs(x_data[i])));
      } else if (label_data[i] < 1.0) {
        y_data[i] = (x_data[i] > 0 ? x_data[i] : 0.0) +
                    log(1.0 + exp(-fabs(x_data[i]))) +
                    (x_data[i] > 0 ? x_data[i] : 0.0) -
                    x_data[i] * label_data[i] +
                    log(1.0 + exp(-fabs(x_data[i])));
      } else {
        y_data[i] = (x_data[i] > 0 ? x_data[i] : 0.0) - x_data[i] +
                    log(1.0 + exp(-fabs(x_data[i]))) +
                    (x_data[i] > 0 ? x_data[i] : 0.0) -
                    x_data[i] * (label_data[i] - 1.0) +
                    log(1.0 + exp(-fabs(x_data[i])));
      }
    }
  }
};

template <typename T>
class TeacherStudentSigmoidLossGradOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* x = context.Input<Tensor>("X");
    const T* x_data = x->data<T>();

    Tensor* dx = context.Output<Tensor>(framework::GradVarName("X"));
    T* dx_data = dx->mutable_data<T>(context.GetPlace());

    const Tensor* labels = context.Input<Tensor>("Label");
    const T* label_data = labels->data<T>();

    T soft_max_up_bound =
        static_cast<T>(context.Attr<float>("soft_max_up_bound"));
    T soft_max_lower_bound =
        static_cast<T>(context.Attr<float>("soft_max_lower_bound"));

    int64_t batch_size = x->dims()[0];

    const framework::Tensor* dOut =
        context.Input<framework::Tensor>(framework::GradVarName("Y"));

    const T* dout_data = dOut->data<T>();

    for (int i = 0; i < batch_size; ++i) {
      T sum_val = x_data[i];
      if (sum_val > soft_max_up_bound) {
        sum_val = soft_max_up_bound;
      } else {
        if (sum_val < soft_max_lower_bound) {
          sum_val = soft_max_lower_bound;
        }
      }

      T pred = 1.0 / (1.0 + exp(-sum_val));
      if (label_data[i] < -1.0) {
        dx_data[i] = 0.0 - pred;
      } else if (label_data[i] < 0.0) {
        dx_data[i] = 1.0 - pred;
      } else {
        dx_data[i] = label_data[i] - 2.0 * pred;
      }
      if (sum_val >= soft_max_up_bound || sum_val <= soft_max_lower_bound) {
        dx_data[i] = 0;
      }
      dx_data[i] *= dout_data[i] * -1;
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(teacher_student_sigmoid_loss,
                  ops::TeacherStudentSigmoidLossOp,
                  ops::TeacherStudentSigmoidLossOpMaker,
                  paddle::framework::DefaultGradOpDescMaker<true>);

REGISTER_OPERATOR(teacher_student_sigmoid_loss_grad,
                  ops::TeacherStudentSigmoidLossGradientOp);

REGISTER_OP_CPU_KERNEL(teacher_student_sigmoid_loss,
                       ops::TeacherStudentSigmoidLossOpKernel<float>,
                       ops::TeacherStudentSigmoidLossOpKernel<double>);

REGISTER_OP_CPU_KERNEL(teacher_student_sigmoid_loss_grad,
                       ops::TeacherStudentSigmoidLossGradOpKernel<float>,
                       ops::TeacherStudentSigmoidLossGradOpKernel<double>);
