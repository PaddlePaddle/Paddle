/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#include "paddle/fluid/operators/fused_softmax_mask_op.h"
#include "paddle/fluid/framework/generator.h"
#include "paddle/fluid/framework/op_registry.h"
namespace paddle {
namespace operators {

using framework::Tensor;

class SoftmaxMaskFuseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "SoftmaxMaskFuse");
    OP_INOUT_CHECK(ctx->HasInput("Mask"), "Input", "Mask", "SoftmaxMaskFuse");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "SoftmaxMaskFuse");
    auto x_dims = ctx->GetInputDim("X");
    auto mask_dims = ctx->GetInputDim("Mask");

    PADDLE_ENFORCE_EQ(
        x_dims.size(), 4,
        platform::errors::InvalidArgument("Input x must be in 4D dimension but "
                                          "received the dimension of X is %d",
                                          x_dims.size()));
    PADDLE_ENFORCE_EQ(mask_dims.size(), 4,
                      platform::errors::InvalidArgument(
                          "Input mask must be in 4D dimension but "
                          "received the dimension of mask is %d",
                          mask_dims.size()));

    ctx->SetOutputDim("Out", x_dims);
    ctx->ShareLoD("X", "Out");
  }
};

class SoftmaxMaskFuseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input of softmax_mask_fuse op, "
             "which is the result of matmul(QK)/sqrt(dk).");
    AddInput("Mask", "The mask attr of the op, multi-head attention's mask");
    AddOutput("Out", "The result of softmax_mask_fuse op.");

    AddComment(R"DOC(
Softmax Mask Fuse Operator.
In general, the compute pass is:
product = matmul(QK)/sqrt(dk)
pre_softmax = product + attn_mask
output = softmax(pre_softmax)
To reduce the launch op time and reduce the number of forward and backward,
and to reduce the memory cost for the pre_softmax var during the compute
this op fuse last two operations into one, so users can simply call
product = matmul(QK)/sqrt(dk)
output = softmax_mask_fuse(product, attn_mask)
to get the final output.
By doing this fusion, we can optimize the training by
1. saving one launch cost, one forward and one backward cost
2. saving the memory cost used to save the tmp var
)DOC");
  }
};

class SoftmaxMaskFuseOpGrad : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")), "Input",
                   framework::GradVarName("Out"), "SoftmaxMaskFuseGrad");

    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    ctx->SetOutputDim(framework::GradVarName("X"), out_dims);
    ctx->ShareLoD(framework::GradVarName("Out"), framework::GradVarName("X"));
  }
};

template <typename T>
class SoftmaxMaskFuseGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("fused_softmax_mask_grad");
    op->SetInput("Softmax", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(fused_softmax_mask, ops::SoftmaxMaskFuseOp,
                  ops::SoftmaxMaskFuseOpMaker,
                  ops::SoftmaxMaskFuseGradOpMaker<paddle::framework::OpDesc>,
                  ops::SoftmaxMaskFuseGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(fused_softmax_mask_grad, ops::SoftmaxMaskFuseOpGrad);
REGISTER_OP_CPU_KERNEL(
    fused_softmax_mask,
    ops::SoftmaxMaskFuseCPUKernel<paddle::platform::CPUDeviceContext, float>,
    ops::SoftmaxMaskFuseCPUKernel<paddle::platform::CPUDeviceContext, double>);
