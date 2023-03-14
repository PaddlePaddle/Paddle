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
#include "paddle/fluid/operators/fused_softmax_mask_upper_triangle_op.h"

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/generator.h"
namespace paddle {
namespace operators {

class SoftmaxMaskFuseUpperTriangleOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(
        ctx->HasInput("X"), "Input", "X", "SoftmaxMaskFuseUpperTriangle");
    OP_INOUT_CHECK(
        ctx->HasOutput("Out"), "Output", "Out", "SoftmaxMaskFuseUpperTriangle");

    auto x_dims = ctx->GetInputDim("X");
    PADDLE_ENFORCE_EQ(
        x_dims.size(),
        4,
        platform::errors::InvalidArgument("Input x must be in 4D dimension but "
                                          "received the dimension of X is %d",
                                          x_dims.size()));

    ctx->SetOutputDim("Out", x_dims);
    ctx->ShareLoD("X", "Out");
  }
};

class SoftmaxMaskFuseUpperTriangleOpMaker
    : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "The input of softmax_mask_fuse_upper_triangle op, "
             "which is the result of matmul(QK)/sqrt(dk).");
    AddOutput("Out", "The result of softmax_mask_fuse_upper_triangle op.");

    AddComment(R"DOC(
Softmax Mask Fuse Operator.
product = matmul(QK)/sqrt(dk)
output = softmax_mask_fuse_upper_triangle(product)
to get the final output.
)DOC");
  }
};

class SoftmaxMaskFuseUpperTriangleOpGrad
    : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   framework::GradVarName("Out"),
                   "SoftmaxMaskFuseUpperTriangleGrad");

    auto out_dims = ctx->GetInputDim(framework::GradVarName("Out"));
    ctx->SetOutputDim(framework::GradVarName("X"), out_dims);
    ctx->ShareLoD(framework::GradVarName("Out"), framework::GradVarName("X"));
  }
};

template <typename T>
class SoftmaxMaskFuseUpperTriangleGradOpMaker
    : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("fused_softmax_mask_upper_triangle_grad");
    op->SetInput("Softmax", this->Output("Out"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    fused_softmax_mask_upper_triangle,
    ops::SoftmaxMaskFuseUpperTriangleOp,
    ops::SoftmaxMaskFuseUpperTriangleOpMaker,
    ops::SoftmaxMaskFuseUpperTriangleGradOpMaker<paddle::framework::OpDesc>,
    ops::SoftmaxMaskFuseUpperTriangleGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(fused_softmax_mask_upper_triangle_grad,
                  ops::SoftmaxMaskFuseUpperTriangleOpGrad);
REGISTER_OP_CPU_KERNEL(
    fused_softmax_mask_upper_triangle,
    ops::SoftmaxMaskFuseUpperTriangleCPUKernel<phi::CPUContext, float>,
    ops::SoftmaxMaskFuseUpperTriangleCPUKernel<phi::CPUContext, double>);
