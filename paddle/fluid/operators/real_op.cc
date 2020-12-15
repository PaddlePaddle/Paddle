/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/real_op.h"

namespace paddle {
namespace operators {

class RealOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "Real");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "Real");

    auto x_dims = ctx->GetInputDim("X");
    ctx->SetOutputDim("Out", x_dims);
    ctx->ShareLoD("X", "Out");
  }
};

class RealOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of real op.");
    AddOutput("Out", "(Tensor), The output tensor of real op.");
    AddComment(R"DOC(
Real Operator.

This operator is used to get a new tensor containing real values
from a tensor with complex data type.

)DOC");
  }
};

template <typename T>
class RealGradOpMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("real");
    grad_op->SetInput("X", this->OutputGrad("Out"));
    grad_op->SetOutput("Out", this->InputGrad("X"));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(real, ops::RealOp, ops::RealOpMaker,
                  ops::RealGradOpMaker<paddle::framework::OpDesc>,
                  ops::RealGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(real, ops::RealKernel<paddle::platform::CPUDeviceContext,
                                             paddle::platform::complex64>,
                       ops::RealKernel<paddle::platform::CPUDeviceContext,
                                       paddle::platform::complex128>);
