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

#include "paddle/fluid/operators/hinge_loss_op.h"

#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace operators {

class HingeLossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Logits"), "Input", "Logits", "HingeLoss");
    OP_INOUT_CHECK(ctx->HasInput("Labels"), "Input", "Labels", "HingeLoss");

    auto pred_dims = ctx->GetInputDim("Logits");
    auto label_dims = ctx->GetInputDim("Labels");

    PADDLE_ENFORCE_EQ(
        pred_dims,
        label_dims,
        platform::errors::InvalidArgument(
            "The Input(input) and Input(label) should have the same "
            "shape, but received input shape [%s] != label shape [%s]",
            pred_dims,
            label_dims));

    PADDLE_ENFORCE_EQ(
        pred_dims.size(),
        2,
        platform::errors::InvalidArgument("Input(input) rank should be 2, "
                                          "but received input rank(%d) != 2",
                                          pred_dims.size()));

    PADDLE_ENFORCE_EQ(pred_dims[1],
                      1,
                      platform::errors::InvalidArgument(
                          "The second dimension of Input(input) should be 1, "
                          "as each row of input contains a real value, "
                          "but received second dimension of input (%d) != 1",
                          pred_dims[1]));

    ctx->SetOutputDim("Loss", {pred_dims[0], 1});
    ctx->ShareLoD("Logits", "Loss");
  }
};

template <typename AttrType>
class HingeLossOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Logits",
             "The input value (Logits) of Hinge loss op."
             "Logits is a 2-D tensor with shape [batch_size, 1].");
    AddInput("Labels",
             "The target value (Labels) of Hinge loss op."
             "Labels is a 2-D tensor with shape [batch_size, 1].");
    AddOutput("Loss",
              "The output tensor with shape [batch_size, 1] "
              "which represents the hinge loss.");
    AddComment(R"DOC(
HingeLoss Operator.

Let x be a logit (prediction) and y be the actual label. The logit can
take any values from (-inf, inf), but the labels should be either -1 or 1.
Then, the hinge loss is computed as follows:

$$
L_(x, y) = max(1 - y.x, 0)
$$

Note that the labels passed as input will have values as either 0 or 1.

)DOC");
  }
};

class HingeLossGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Logits"), "Input", "Logits", "HingeLossGrad");
    OP_INOUT_CHECK(ctx->HasInput("Labels"), "Input", "Labels", "HingeLossGrad");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Loss")),
                   "Input",
                   "Loss@GRAD",
                   "HingeLossGrad");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("Logits")),
                   "Output",
                   "Logits@GRAD",
                   "HingeLossGrad");

    auto pred_dims = ctx->GetInputDim("Logits");
    auto loss_grad_dims = ctx->GetInputDim(framework::GradVarName("Loss"));

    PADDLE_ENFORCE_EQ(loss_grad_dims,
                      pred_dims,
                      platform::errors::InvalidArgument(
                          "The shape of loss gradient should be the same as "
                          "the shape of Input(input), but received the loss "
                          "gradient shape [%s] != input shape [%s]",
                          loss_grad_dims,
                          pred_dims));

    auto pred_grad_name = framework::GradVarName("Logits");
    ctx->SetOutputDim(pred_grad_name, pred_dims);
  }
};

template <typename T>
class HingeLossGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("hinge_loss_grad");
    op->SetInput("Logits", this->Input("Logits"));
    op->SetInput("Labels", this->Input("Labels"));
    op->SetInput(framework::GradVarName("Loss"), this->OutputGrad("Loss"));
    op->SetOutput(framework::GradVarName("Logits"), this->InputGrad("Logits"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(hinge_loss,
                  ops::HingeLossOp,
                  ops::HingeLossOpMaker<float>,
                  ops::HingeLossGradOpMaker<paddle::framework::OpDesc>,
                  ops::HingeLossGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(hinge_loss_grad, ops::HingeLossGradOp);
REGISTER_OP_CPU_KERNEL(hinge_loss,
                       ops::HingeLossKernel<phi::CPUContext, float>);
REGISTER_OP_CPU_KERNEL(hinge_loss_grad,
                       ops::HingeLossGradKernel<phi::CPUContext, float>);

REGISTER_OP_CUDA_KERNEL(hinge_loss,
                        ops::HingeLossKernel<phi::GPUContext, float>);
REGISTER_OP_CUDA_KERNEL(hinge_loss_grad,
                        ops::HingeLossGradKernel<phi::GPUContext, float>);
