/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/diag_op.h"

namespace paddle {
namespace operators {

class DiagOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Diagonal"), "Input", "Diagonal", "diag");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "diag");

    auto s_dims = ctx->GetInputDim("Diagonal");

    PADDLE_ENFORCE_EQ(
        s_dims.size(),
        1UL,
        platform::errors::InvalidArgument(
            "The dimension of 'diagonal' must be 1, but now it is %d.",
            s_dims.size()));

    ctx->SetOutputDim("Out", {s_dims[0], s_dims[0]});
  }
};

class DiagOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Diagonal",
             "Diagonal values of square matrix. It is a tensor with rank 1.");
    AddOutput("Out", "A square matrix.");
    AddComment(R"DOC(
    Return a square matrix with specified diagonal values.
)DOC");
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    diag,
    ops::DiagOp,
    ops::DiagOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
REGISTER_OP_CPU_KERNEL(diag,
                       ops::DiagKernel<phi::CPUContext, int>,
                       ops::DiagKernel<phi::CPUContext, float>,
                       ops::DiagKernel<phi::CPUContext, double>,
                       ops::DiagKernel<phi::CPUContext, int64_t>);
