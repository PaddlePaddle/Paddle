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

#include "paddle/fluid/operators/triplet_loss_op.h"

namespace paddle {
namespace operators {

std::vector<int> GetOffsets<platform::CPUDeviceContext>(const Tensor* t) {
    std::vector<int> offsets;
    int64_t* data = t->data<int64_t>();
    int offset = 0;
    int64_t currrent_value = data[0];
    for (int i=1; i<t->numel(); ++i) {
        if (data[i] != currrent_value) {
            offsets.push(i);
        }
        currrent_value = data[i]; 
    }
    offsets.push(t->numel());
}

class TripletLossOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Logits",
             "(Tensor, default: Tensor<float>), The unscaled log probabilities "
             "which is a 2-D tensor with shape [N x K]. N is the batch_size, "
             "and K is the class number.");
    AddInput("Label",
             "(Tensor) The ground truth which is a 2-D tensor. If soft_label "
             "is set to false, Label is a Tensor<int64> with shape [N x 1]. If "
             "soft_label is set to true, Label is a Tensor<float/double> with "
             "shape [N x K].");
    AddOutput("Loss",
              "(Tensor, default: Tensor<float>), A 2-D tensor. The cross "
              "entropy loss with shape [N x 1].");
    AddComment(R"DOC(

)DOC");
  }
};

class TripletLossOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("Logits"),
                   "Input(Logits) should be not null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput("Loss"), "Output(Loss) should be not null.");
    auto labels_dims = ctx->GetInputDim("Label");
    PADDLE_ENFORCE_EQ(
        logits_dims.size(), 2UL,
        "The input of triplet_loss should be a 2-D tensor.");
    PADDLE_ENFORCE_EQ(labels_dims.size(), 2UL,
                      "The labels should be a 2-D tensor.");
    PADDLE_ENFORCE_EQ(labels_dims[1], 1UL,
                        "The 2nd dimension of "
                        "Input(Label) should be 1.");
    ctx->SetOutputDim("Loss", {logits_dims[0], 1});
    ctx->ShareLoD("Logits", /*->*/ "Loss");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(ctx.Input<Tensor>("Logits")->type()),
        ctx.device_context());
  }
};

class TripletLossOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Loss")),
                   "Input(Loss@Grad) should not be null.");
    PADDLE_ENFORCE(ctx->HasInput("Label"), "Input(Label) should be not null.");
    PADDLE_ENFORCE(ctx->HasOutput(framework::GradVarName("Logits")),
                   "Output(Logits@Grad) should be not null.");

    auto labels_dims = ctx->GetInputDim("Label");
    PADDLE_ENFORCE_EQ(labels_dims.size(), 2UL,
                      "The labels should be a 2-D tensor.");

    PADDLE_ENFORCE_EQ(labels_dims[1], 1UL,
                        "When Attr(soft_label) == false, the 2nd dimension of "
                        "Input(Label) should be 1.");

    ctx->SetOutputDim(framework::GradVarName("Logits"),
                      ctx->GetInputDim("Softmax"));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return framework::OpKernelType(
        framework::ToDataType(
            ctx.Input<Tensor>(framework::GradVarName("Loss"))->type()),
        ctx.device_context());
  }
};


}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(triplet_loss, ops::TripletLossOp,
                  ops::TripletLossOpMaker);

REGISTER_OP_CPU_KERNEL(triplet_loss,
                       ops::TripletLossKernel<platform::CPUDeviceContext, float>,
                       ops::TripletLossKernel<platform::CPUDeviceContext, double>);
