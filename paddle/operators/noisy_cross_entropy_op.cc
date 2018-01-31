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

#include "paddle/operators/noisy_cross_entropy_op.h"
#include "paddle/operators/math/math_function.h"
#include <iostream>

namespace paddle {
namespace operators {

class NoisyCrossEntropyOp : public framework::OperatorWithKernel {
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
  // Explicitly set that the data type of computation kernel of cross_entropy
  // is determined by its input "X".
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("X")->type()),
        ctx.device_context());
  }
};

class NoisyCrossEntropyGradientOp : public framework::OperatorWithKernel {
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
  // Explicitly set that the data type of computation kernel of cross_entropy
  // is determined by its input "X".
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("X")->type()),
        ctx.device_context());
  }
};

class NoisyCrossEntropyOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  NoisyCrossEntropyOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X",
             "(Tensor, default Tensor<float>), a 2-D tensor with shape [N x D],"
             " where N is the batch size and D is the number of classes. "
             "This input is a probability computed by the previous operator, "
             "which is almost always the result of a softmax operator.");
    AddInput("Label",
             "(Tensor), the ground truth which is a 2-D tensor. "
             "Label is a Tensor<int64> with shape [N x 1]. ");
    AddOutput("Y",
              "(Tensor, default Tensor<float>), a 2-D tensor with shape "
              "[N x 1]. The cross entropy loss.");
    AddAttr<float>("noise",
                  "fp32, default 0.1")
        .SetDefault(0.1);
    AddComment(R"DOC(
NoisyCrossEntropy Operator.

It supports both standard cross-entropy with label smoothing computation.
label smooth cross-entropy: Label[i, 0] indicates the class index for sample i:
    Noisy_label[i, j] = noise / (class_num - 1) if j != Label[i, 0]
    Noisy_label[i, j] = 1.0 - noise if j == Label[i, 0]
    $Y[i] = \sum_j{-Noisy_label[i, j] * log(X[i, j])}$

Both the input X and Label can carry the LoD (Level of Details) information,
or not. But the output only shares the LoD information with input X.
)DOC");
  }
};

template <typename T>
class NoisyCrossEntropyOpKernel :
        public framework::OpKernel<T> {
public:
    void Compute(const framework::ExecutionContext& context)
        const override {
        PADDLE_ENFORCE(platform::is_cpu_place(context.GetPlace()),
                       "This kernel only runs on CPU.");
        Tensor* y = context.Output<Tensor>("Y");
        const Tensor* x = context.Input<Tensor>("X");
        const Tensor* labels = context.Input<Tensor>("Label");
        T* y_data = y->mutable_data<T>(context.GetPlace());
        const T* x_data = x->data<T>();
        const int64_t* label_data = labels->data<int64_t>();
        int64_t c_num = x->dims()[1];
        T noise = static_cast<T>(context.Attr<float>("noise"));
        T neg_noise = noise / (static_cast<T>(c_num) - 1.0);
        int64_t batch_size = x->dims()[0];
        for (int i = 0; i < batch_size; ++i) {
            y_data[i] = 0.0;
            int64_t offset = i * c_num;
            for (int64_t j = 0; j < c_num; ++j) {
                if (j == label_data[i]) {
                    y_data[i] += -(1.0 - noise) * logf(x_data[offset + j]);
                } else {
                    y_data[i] += -neg_noise * logf(x_data[offset + j]);
                }
            }
        }
    }
};

template <typename T>
class NoisyCrossEntropyGradOpKernel :
        public framework::OpKernel<T> {
public:
    void Compute(const framework::ExecutionContext& context)
        const override {
        const Tensor* labels = context.Input<Tensor>("Label");
        const Tensor* x = context.Input<Tensor>("X");
        const Tensor* dy = context.Input<Tensor>(framework::GradVarName("Y"));
        Tensor * dx = context.Output<Tensor>(framework::GradVarName("X"));
        T* dx_data = dx->mutable_data<T>(context.GetPlace());
        const T* dy_data = dy->data<T>();
        const T * x_data = x->data<T>();
        const int64_t* label_data = labels->data<int64_t>();
        int64_t batch_size = x->dims()[0];
        int64_t c_num = x->dims()[1];
        T noise = static_cast<T>(context.Attr<float>("noise"));
        T neg_noise = noise / (c_num - 1);
        for (int64_t i = 0; i < batch_size; ++i) {
            for (int64_t j = 0; j < c_num; ++j) {
                if (j == label_data[i]) {
                    dx_data[i * c_num + j] = -dy_data[i] * (1.0 - noise) /
                        x_data[i * c_num + j];
                } else {
                    dx_data[i * c_num + j] = -dy_data[i] * neg_noise /
                        x_data[i * c_num + j];
                }
            }
        }
    }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP(noisy_cross_entropy,
            ops::NoisyCrossEntropyOp,
            ops::NoisyCrossEntropyOpMaker,
            noisy_cross_entropy_grad,
            ops::NoisyCrossEntropyGradientOp);
REGISTER_OP_CPU_KERNEL(noisy_cross_entropy,
                       ops::NoisyCrossEntropyOpKernel<float>,
                       ops::NoisyCrossEntropyOpKernel<double>);
REGISTER_OP_CPU_KERNEL(noisy_cross_entropy_grad,
                       ops::NoisyCrossEntropyGradOpKernel<float>,
                       ops::NoisyCrossEntropyGradOpKernel<double>);
