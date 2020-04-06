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

#include "paddle/fluid/operators/cholesky_op.h"

namespace paddle {
namespace operators {

using framework::OpKernelType;
using framework::Tensor;

class CholeskyOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE_EQ(ctx->HasInput("X"), true,
                      platform::errors::NotFound(
                          "Input(X) of CholeskyOp should not be null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::NotFound(
                          "Output(Out) of CholeskyOp should not be null."));

    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
  }
};

class CholeskyOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(Tensor), The input tensor of cholesky op. Its shape should be "
             "[*, M, M] where * is zero or more batch dimensions, and matrices "
             "on the inner-most 2 dimensions all should be symmetric "
             "positive-definite.");
    AddOutput("Out",
              "(Tensor), The output tensor of cholesky op. It has the same "
              "shape as the input, and it is composed of upper-triangular or "
              "lower-triangular Cholesky factors of each of the individual "
              "matrices.");
    AddAttr<bool>("upper",
                  "(bool, default false), flag indicating whether to return "
                  "upper or lower triangular matrices. Default: False")
        .SetDefault(false);
    AddComment(R"DOC(
Cholesky Operator.

Computes the Cholesky decomposition of one symmetric positive-definite matrix
or batches of symmetric positive-definite matrices.

)DOC");
  }
};

class CholeskyGradOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) should not be null");
    PADDLE_ENFORCE(ctx->HasInput(framework::GradVarName("Out")),
                   "Input(Out@GRAD) should not be null");
    auto x_dims = ctx->GetInputDim("X");
    auto x_grad_name = framework::GradVarName("X");
    if (ctx->HasOutput(x_grad_name)) {
      ctx->SetOutputDim(x_grad_name, x_dims);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    cholesky, ops::CholeskyOp, ops::CholeskyOpMaker,
    paddle::framework::DefaultGradOpMaker<paddle::framework::OpDesc, true>,
    paddle::framework::DefaultGradOpMaker<paddle::imperative::OpBase, true>);
REGISTER_OPERATOR(cholesky_grad, ops::CholeskyGradOp);

REGISTER_OP_CPU_KERNEL(cholesky, ops::CholeskyCPUKernel<float>,
                       ops::CholeskyCPUKernel<double>);

REGISTER_OP_CPU_KERNEL(cholesky_grad, ops::CholeskyGradCPUKernel<float>,
                       ops::CholeskyGradCPUKernel<double>);
