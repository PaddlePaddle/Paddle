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

#include "paddle/fluid/operators/l1_norm_op.h"

#include <memory>

namespace paddle {
namespace operators {

class L1NormOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "L1NormOp");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "L1NormOp");

    ctx->SetOutputDim("Out", {1});
  }
};

class L1NormGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "L1NormGradOp");
    OP_INOUT_CHECK(ctx->HasInput(framework::GradVarName("Out")),
                   "Input",
                   "Out@GRAD",
                   "L1NormGradOp");
    OP_INOUT_CHECK(ctx->HasOutput(framework::GradVarName("X")),
                   "Output",
                   "X@GRAD",
                   "L1NormGradOp");

    ctx->SetOutputDim(framework::GradVarName("X"), ctx->GetInputDim("X"));
  }
};

class L1NormOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The input of l1_norm op.");
    AddOutput("Out", "(Scalar) The output of l1_norm op.");
    AddComment(R"DOC(
L1 Norm Operator.

Computes the L1 norm of a tensor.

$$Out = \sum{|X|}$$

)DOC");
  }
};

template <typename T>
class L1NormGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> op) const override {
    op->SetType("l1_norm_grad");
    op->SetInput("X", this->Input("X"));
    op->SetInput(framework::GradVarName("Out"), this->OutputGrad("Out"));
    op->SetOutput(framework::GradVarName("X"), this->InputGrad("X"));
    op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(l1_norm,
                  ops::L1NormOp,
                  ops::L1NormOpMaker,
                  ops::L1NormGradMaker<paddle::framework::OpDesc>,
                  ops::L1NormGradMaker<paddle::imperative::OpBase>);
REGISTER_OPERATOR(l1_norm_grad, ops::L1NormGradOp);
REGISTER_OP_CPU_KERNEL(l1_norm, ops::L1NormKernel<phi::CPUContext, float>);
REGISTER_OP_CPU_KERNEL(l1_norm_grad,
                       ops::L1NormGradKernel<phi::CPUContext, float>);

REGISTER_OP_CUDA_KERNEL(l1_norm, ops::L1NormKernel<phi::GPUContext, float>);
REGISTER_OP_CUDA_KERNEL(l1_norm_grad,
                        ops::L1NormGradKernel<phi::GPUContext, float>);
